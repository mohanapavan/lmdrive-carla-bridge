"""
CARLA-facing LMDrive agent (Python 3.7, official CARLA 0.9.10.1 client API).

This is lmdriver_agent.py with the torch model replaced by a socket call to
inference_server.py, which runs in a separate Python 3.10 process holding
the GPU model (this machine's GPU has no kernels for the torch version this
repo pins, which only ships py3.5-3.7 wheels anyway -- so the model can't
live in this process).
"""
import os
import json
import random
import datetime
import pathlib
import time
import imp
import socket
import struct
import pickle
import math

import cv2
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner, InstructionPlanner
from team_code.pid_controller import PIDController

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", None)
INFER_HOST = os.environ.get("INFER_HOST", "127.0.0.1")
INFER_PORT = int(os.environ.get("INFER_PORT", "8765"))


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def lidar_to_raw_features(lidar):
    def preprocess(lidar_xyzr, lidar_painted=None):
        idx = (lidar_xyzr[:, 0] > -1.2) & (lidar_xyzr[:, 0] < 1.2) & (lidar_xyzr[:, 1] > -1.2) & (lidar_xyzr[:, 1] < 1.2)
        idx = np.argwhere(idx)
        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    lidar_xyzr = preprocess(lidar)
    idxs = np.arange(len(lidar_xyzr))
    np.random.shuffle(idxs)
    lidar_xyzr = lidar_xyzr[idxs]

    out = np.zeros((40000, 4), dtype=np.float32)
    num_points = min(40000, len(lidar_xyzr))
    out[:num_points, :4] = lidar_xyzr
    out[np.isinf(out)] = 0
    out[np.isnan(out)] = 0
    out = rotate_lidar(out, -90).astype(np.float32)
    return out, num_points


class InferenceClient:
    """Persistent socket connection to inference_server.py."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=30.0)

    def _send(self, obj):
        body = pickle.dumps(obj, protocol=4)
        self.sock.sendall(struct.pack(">I", len(body)) + body)

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("inference server closed connection")
            buf += chunk
        return buf

    def _recv(self):
        (length,) = struct.unpack(">I", self._recv_exact(4))
        return pickle.loads(self._recv_exact(length))

    def call(self, obj):
        self._send(obj)
        resp = self._recv()
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError("inference server error: " + resp["error"])
        return resp


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 900
        self._surface = None
        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("LMDrive Agent (bridge)")

    def run_interface(self, input_data):
        rgb = input_data["rgb_front"]
        surface = np.zeros((900, 1200, 3), np.uint8)
        surface[:, :1200] = rgb
        surface[:210, :280] = input_data["rgb_left"]
        surface[:210, 920:1200] = input_data["rgb_right"]
        surface[:210, 495:705] = input_data["rgb_center"]
        surface = cv2.putText(surface, input_data["time"], (20, 710), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data["meta_control"], (20, 740), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data["waypoints"], (20, 770), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data["instruction"], (20, 800), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data["notice"], (20, 830), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        surface[:210, 278:282] = [139, 69, 19]
        surface[:210, 493:497] = [139, 69, 19]
        surface[:210, 703:707] = [139, 69, 19]
        surface[:210, 918:922] = [139, 69, 19]
        surface[208:212, :280] = [139, 69, 19]
        surface[208:212, 920:1200] = [139, 69, 19]
        surface[208:212, 495:705] = [139, 69, 19]

        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "LMDriveBridgeAgent"


class LMDriveBridgeAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self._hic = DisplayInterface()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()

        self.turn_controller = PIDController(K_P=self.config.turn_KP, K_I=self.config.turn_KI, K_D=self.config.turn_KD, n=self.config.turn_n)
        self.speed_controller = PIDController(K_P=self.config.speed_KP, K_I=self.config.speed_KI, K_D=self.config.speed_KD, n=self.config.speed_n)

        self.agent_use_notice = self.config.agent_use_notice
        self.sample_rate = self.config.sample_rate * 2  # CARLA runs at 20Hz

        print("connecting to inference server at %s:%d ..." % (INFER_HOST, INFER_PORT))
        self.infer = InferenceClient(INFER_HOST, INFER_PORT)
        self.infer.call({"cmd": "ping"})
        self.infer.call({"cmd": "reset_episode", "sample_rate": self.sample_rate})
        print("connected.")

        self.curr_instruction = "Drive safely."
        self.curr_notice = ""
        self.active_misleading_instruction = False
        self.remaining_misleading_frames = 0
        self.buffer_len = 0

        self.prev_lidar = None
        self.prev_control = None
        self.sampled_scenarios = None
        self.instruction = ""

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(map(lambda x: "%02d" % x, (now.month, now.day, now.hour, now.minute, now.second)))
            print("Data save path:", string)
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(5, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self._instruction_planner = InstructionPlanner(self.scenario_cofing_name, True)
        self.initialized = True
        random.seed("".join([str(x[0]) for x in self._global_plan]))

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": 1200, "height": 900, "fov": 100, "id": "rgb_front"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},
            {"type": "sensor.camera.rgb", "x": -1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 180.0, "width": 400, "height": 300, "fov": 100, "id": "rgb_rear"},
            {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):
        rgb_front = cv2.cvtColor(input_data["rgb_front"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data["rgb_rear"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0

        result = {"rgb_front": rgb_front, "rgb_left": rgb_left, "rgb_right": rgb_right, "rgb_rear": rgb_rear, "gps": gps, "speed": speed, "compass": compass}
        pos = self._get_position(result)

        lidar_data = input_data["lidar"][1]
        lidar_unprocessed = lidar_data[..., :4]
        if self.prev_lidar is not None:
            lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar])
        else:
            lidar_unprocessed_full = lidar_unprocessed
        self.prev_lidar = lidar_unprocessed

        lidar_processed, num_points = lidar_to_raw_features(lidar_unprocessed_full)
        result["lidar"] = lidar_processed
        result["num_points"] = num_points
        result["gps"] = pos

        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_waypoint"] = next_wp
        result["next_command"] = next_cmd.value
        result["measurements"] = [pos[0], pos[1], compass, speed]
        result["speed"] = speed

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point
        return result

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        self.step += 1
        tick_data = self.tick(input_data)

        if self.step < 20:
            control = carla_control(steer=0.0, throttle=0.0, brake=1.0)
            return control

        if self.step % 2 != 0 and self.step > 4:
            return self.prev_control

        velocity = tick_data["speed"]

        last_instruction = self._instruction_planner.command2instruct(self.town_id, tick_data, self._route_planner.route)
        last_notice = self._instruction_planner.pos2notice(self.sampled_scenarios, tick_data)
        last_traffic_light_notice = self._instruction_planner.traffic_notice(tick_data)
        last_misleading_instruction = self._instruction_planner.command2mislead(self.town_id, tick_data)
        if last_notice == "":
            last_notice = last_traffic_light_notice

        reset_buffer = False
        if self.curr_instruction != last_instruction or self.buffer_len > 400:
            if self.remaining_misleading_frames > 0:
                self.remaining_misleading_frames -= 1
            else:
                self.active_misleading_instruction = False
                if last_misleading_instruction != "" and random.random() < 0.2:
                    self.curr_instruction = last_misleading_instruction
                    self.active_misleading_instruction = True
                    self.remaining_misleading_frames = 20
                else:
                    self.curr_instruction = last_instruction
                reset_buffer = True
                self.buffer_len = 0
                self.curr_notice = ""

        if last_notice != "" and last_notice != self.curr_notice:
            self.curr_notice = last_notice
            notice_frame_id = self.buffer_len
        else:
            notice_frame_id = -1

        rgb_center = cv2.resize(tick_data["rgb_front"], (800, 600))

        req = {
            "reset_buffer": reset_buffer,
            "rgb_front": tick_data["rgb_front"],
            "rgb_left": tick_data["rgb_left"],
            "rgb_right": tick_data["rgb_right"],
            "rgb_rear": tick_data["rgb_rear"],
            "rgb_center": rgb_center,
            "target_point": tick_data["target_point"],
            "lidar": tick_data["lidar"],
            "num_points": tick_data["num_points"],
            "velocity": tick_data["speed"],
            "instruction_text": self.curr_instruction,
            "use_notice": self.agent_use_notice,
            "notice_text": self.curr_notice,
            "notice_frame_id": notice_frame_id,
        }
        resp = self.infer.call(req)
        self.buffer_len += 1
        waypoints = resp["waypoints"]
        end_prob = resp["end_prob"]

        steer, throttle, brake, metadata = self.control_pid(waypoints, velocity)

        if end_prob > 0.75:
            self.buffer_len = 0
            self.curr_notice = ""

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla_control(steer=float(steer) * 0.8, throttle=float(throttle), brake=float(brake))

        display_data = {}
        display_data["rgb_front"] = cv2.resize(tick_data["rgb_front"], (1200, 900))
        display_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (280, 210))
        display_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (280, 210))
        display_data["rgb_center"] = cv2.resize(tick_data["rgb_front"][330:570, 480:720], (210, 210))
        display_data["instruction"] = ("Instruction: [Misleading] %s" if self.active_misleading_instruction else "Instruction: %s") % self.curr_instruction
        display_data["time"] = "Time: %.3f. Frames: %d. End prob: %.2f" % (timestamp, self.buffer_len, end_prob)
        display_data["meta_control"] = "Throttle: %.2f. Steer: %.2f. Brake: %.2f" % (control.steer, control.throttle, control.brake)
        display_data["waypoints"] = "Waypoints: (%.1f, %.1f), (%.1f, %.1f)" % (waypoints[0, 0], -waypoints[0, 1], waypoints[1, 0], -waypoints[1, 1])
        display_data["notice"] = "Notice: %s" % last_notice
        surface = self._hic.run_interface(display_data)
        tick_data["surface"] = surface

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control

        if SAVE_PATH is not None:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step - 20
        Image.fromarray(tick_data["surface"]).save(self.save_path / "meta" / ("%04d.jpg" % frame))

    def destroy(self):
        pass

    def control_pid(self, waypoints, velocity):
        assert waypoints.shape[0] == 5
        waypoints = np.array(waypoints, dtype=np.float64)
        waypoints[:, 1] *= -1
        speed = velocity

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if speed < 0.01:
            angle = np.array(0.0)
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            "speed": float(speed), "steer": float(steer), "throttle": float(throttle), "brake": float(brake),
            "desired_speed": float(desired_speed), "angle": float(angle),
        }
        return steer, throttle, brake, metadata


def carla_control(steer, throttle, brake):
    import carla
    control = carla.VehicleControl()
    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    return control
