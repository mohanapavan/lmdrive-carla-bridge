import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LAVIS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vision_encoder"))

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from lavis.common.registry import registry
import lavis.models.drive_models.drive  # noqa: register vicuna_drive

WEIGHTS = "/home/user/.config/Claude/scratch-workspaces/0267dccf-48a8-4f04-9830-06429306d176/60800dfe-88a3-486a-b2ac-bc8533c8b9c8/scratch-2026-10-01-1b8572/carla_vla_project/weights"

model_cls = registry.get_model_class('vicuna_drive')
print("building model...")
model = model_cls(
    preception_model="memfuser_baseline_e1d3_return_feature",
    preception_model_ckpt=f"{WEIGHTS}/vision-encoder-r50/vision-encoder-r50.pth.tar",
    llm_model=f"{WEIGHTS}/llava-v1.5-7b-base",
    max_txt_len=64,
    use_notice_prompt=False,
)
print("model built OK:", type(model))

print("loading lmdrive checkpoint...")
sd = torch.load(f"{WEIGHTS}/lmdrive-llava-v1.5-7b/llava-v1.5-checkpoint.pth", map_location="cpu")
missing, unexpected = model.load_state_dict(sd["model"], strict=False)
print("load_state_dict done. missing:", len(missing), "unexpected:", len(unexpected))

model = model.half().cuda()
model.eval()
print("MODEL ON GPU OK")
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
