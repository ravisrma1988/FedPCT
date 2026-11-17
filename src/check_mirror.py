"""Compare the HF mirror emmilly/iwildcam-wilds-v2 with the official WILDS iWildCam metadata."""
import inspect
import re

from huggingface_hub import HfApi
from wilds.datasets import iwildcam_dataset

src = inspect.getsource(iwildcam_dataset)
m = re.search(r"_versions_dict\s*=\s*\{.*?\n    \}", src, re.S)
print("WILDS official:", m.group(0) if m else "not found")

api = HfApi()
info = api.dataset_info("emmilly/iwildcam-wilds-v2", files_metadata=True)
print("\nHF repo created:", info.created_at, "last modified:", info.last_modified, "author:", info.author)
for f in info.siblings:
    sha = f.lfs.sha256 if f.lfs else None
    print(f"  {f.rfilename}  size={f.size}  lfs_sha256={sha}")
