"""Smoke test: load BioCLIP on the MIG slice and time image encoding."""
import time

import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
model = model.cuda().eval()
x = torch.randn(256, 3, 224, 224, device="cuda")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    model.encode_image(x)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        f = model.encode_image(x)
    torch.cuda.synchronize()

dt = (time.time() - t) / 5
print(f"feature shape {tuple(f.shape)}, {256 / dt:.0f} img/s, "
      f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
