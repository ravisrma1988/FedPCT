"""Print open_clip's VisionTransformer.forward and BioCLIP-2's tower config, to split blocks exactly."""
import inspect

import open_clip
import torch

model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
v = model.visual
print(type(v).__name__, "| blocks:", len(v.transformer.resblocks), "| width:", v.transformer.width,
      "| attn_pool:", getattr(v, "attn_pool", None) is not None, "| pool_type:", getattr(v, "pool_type", None),
      "| final_ln_after_pool:", getattr(v, "final_ln_after_pool", None),
      "| proj:", None if v.proj is None else tuple(v.proj.shape))
print("batch_first:", getattr(v.transformer, "batch_first", None))
print(inspect.getsource(type(v).forward))
print(inspect.getsource(type(v.transformer).forward))
print("params total (M):", sum(p.numel() for p in v.parameters()) / 1e6,
      "| last 2 blocks + ln_post + proj (M):",
      (sum(p.numel() for b in v.transformer.resblocks[-2:] for p in b.parameters())
       + sum(p.numel() for p in v.ln_post.parameters()) + v.proj.numel()) / 1e6)
