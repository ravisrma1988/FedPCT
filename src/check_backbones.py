"""What is available for a GPU-bound fine-tuning pipeline?"""
import importlib.util as u
import os

import torch

print("DALI:", bool(u.find_spec("nvidia.dali")), "| torchvision.io.decode_jpeg on cuda:",
      hasattr(__import__("torchvision").io, "decode_jpeg"))
print("CPU cores visible:", os.cpu_count(), "| GPU:", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"GPU mem free/total: {free/2**30:.1f}/{total/2**30:.1f} GB")

from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
for repo in ["imageomics/bioclip", "imageomics/bioclip-2"]:
    try:
        info = api.model_info(repo)
        size = sum((s.size or 0) for s in api.model_info(repo, files_metadata=True).siblings)
        print(f"{repo}: exists, last modified {info.last_modified}, files {size/2**30:.2f} GB, "
              f"tags {[t for t in info.tags if 'license' in t or 'clip' in t][:5]}")
    except Exception as e:  # noqa: BLE001
        print(f"{repo}: not available ({type(e).__name__})")
