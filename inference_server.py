"""
GPU-side inference server for LMDrive.

Runs in the Python 3.10 / modern-torch conda env (needed for this machine's
RTX 5080, which has no kernels under the old torch==2.0.1 this repo pins).
The CARLA client itself only has an official Python 3.7 binding for version
0.9.10.1, so the sim-facing side runs in a separate Python 3.7 process
(lmdrive_bridge_agent.py) and talks to this process over a local TCP socket.

Wire format: 4-byte big-endian length prefix + pickle (protocol 4, readable
by both py3.7 and py3.10).
"""
import os
import sys
import socket
import struct
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LAVIS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vision_encoder"))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from lavis.common.registry import registry
import lavis.models.drive_models.drive  # noqa: registers 'vicuna_drive'

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights")
WEIGHTS = os.environ.get("LMDRIVE_WEIGHTS", WEIGHTS)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        return pil_img.resize(self.size)


def create_carla_rgb_transform(input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    img_size = input_size[-2:] if isinstance(input_size, (tuple, list)) else input_size
    input_size_num = input_size[-1] if isinstance(input_size, (tuple, list)) else input_size
    tfl = []
    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    return transforms.Compose(tfl)


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    hdr = recv_exact(conn, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack(">I", hdr)
    body = recv_exact(conn, length)
    if body is None:
        return None
    return pickle.loads(body)


def send_msg(conn, obj):
    body = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(body)) + body)


class EpisodeState:
    def __init__(self, sample_rate):
        self.visual_feature_buffer = []
        self.curr_notice = ""
        self.curr_notice_frame_id = -1
        self.sample_rate = sample_rate

    def update_and_collect(self, image_embeds):
        self.visual_feature_buffer.append(image_embeds)
        result = self.visual_feature_buffer[:: self.sample_rate]
        if (len(self.visual_feature_buffer) - 1) % self.sample_rate != 0:
            result.append(self.visual_feature_buffer[-1])
        return torch.stack(result, 1)


def build_model():
    os.environ["LMDRIVE_LLM_8BIT"] = "1"
    model_cls = registry.get_model_class("vicuna_drive")
    print("[infer] building model (LLM in 8-bit)...", flush=True)
    model = model_cls(
        preception_model="memfuser_baseline_e1d3_return_feature",
        preception_model_ckpt=f"{WEIGHTS}/vision-encoder-r50/vision-encoder-r50.pth.tar",
        llm_model=f"{WEIGHTS}/llava-v1.5-7b-base",
        max_txt_len=64,
        use_notice_prompt=False,
        split_section_num_for_visual_encoder=8,
    )
    print("[infer] loading lmdrive checkpoint...", flush=True)
    sd = torch.load(f"{WEIGHTS}/lmdrive-llava-v1.5-7b/llava-v1.5-checkpoint.pth", map_location="cpu")

    # llm_model is already int8-quantized and placed on GPU via device_map;
    # detach it before the blanket .half().cuda() so we don't corrupt it.
    llm = model.llm_model
    model.llm_model = None
    model.load_state_dict(sd["model"], strict=False)
    model = model.half().cuda()
    model.llm_model = llm
    model.eval()
    print("[infer] model ready. GPU mem allocated (GB):", torch.cuda.memory_allocated() / 1e9, flush=True)
    return model


@torch.no_grad()
def run_inference(model, transforms_, state, req):
    if req.get("reset_buffer"):
        state.visual_feature_buffer = []
        state.curr_notice = req.get("notice_text", "")
        state.curr_notice_frame_id = req.get("notice_frame_id", -1)

    t_front, t_left, t_right, t_center = transforms_

    def to_tensor(arr, tf):
        return tf(Image.fromarray(arr)).unsqueeze(0).cuda().half()

    rgb_front = to_tensor(req["rgb_front"], t_front)
    rgb_left = to_tensor(req["rgb_left"], t_left)
    rgb_right = to_tensor(req["rgb_right"], t_right)
    rgb_rear = to_tensor(req["rgb_rear"], t_right)
    rgb_center = to_tensor(req["rgb_center"], t_center)

    input_data = {
        "rgb_front": rgb_front,
        "rgb_left": rgb_left,
        "rgb_right": rgb_right,
        "rgb_center": rgb_center,
        "rgb_rear": rgb_rear,
        "target_point": torch.tensor(req["target_point"]).cuda().view(1, 2).half(),
        "lidar": torch.from_numpy(req["lidar"]).half().cuda().unsqueeze(0),
        "num_points": torch.tensor([req["num_points"]]).cuda().unsqueeze(0),
        "velocity": torch.tensor([req["velocity"]]).cuda().view(1, 1).half(),
        "text_input": [req["instruction_text"]],
    }

    image_embeds = model.visual_encoder(input_data)
    image_embeds = state.update_and_collect(image_embeds)
    input_data["valid_frames"] = [image_embeds.size(1)]

    if req.get("use_notice"):
        input_data["notice_text"] = [state.curr_notice]
        input_data["notice_frame_id"] = [state.curr_notice_frame_id]

    waypoints, is_end = model(input_data, inference_mode=True, image_embeds=image_embeds)

    waypoints = waypoints[-1].view(5, 2).float().cpu().numpy()
    end_prob = torch.nn.functional.softmax(is_end, dim=1)[-1][1].float().item()
    return {"waypoints": waypoints, "end_prob": end_prob}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    model = build_model()
    transforms_ = (
        create_carla_rgb_transform(224),
        create_carla_rgb_transform(128),
        create_carla_rgb_transform(128),
        create_carla_rgb_transform(128, need_scale=False),
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[infer] READY listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        print("[infer] client connected:", addr, flush=True)
        state = EpisodeState(sample_rate=4)  # GlobalConfig.sample_rate(2) * 2
        try:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                if req.get("cmd") == "ping":
                    send_msg(conn, {"ok": True})
                    continue
                if req.get("cmd") == "reset_episode":
                    state = EpisodeState(sample_rate=req.get("sample_rate", 4))
                    send_msg(conn, {"ok": True})
                    continue
                resp = run_inference(model, transforms_, state, req)
                send_msg(conn, resp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                send_msg(conn, {"error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()
            print("[infer] client disconnected", flush=True)


if __name__ == "__main__":
    main()
