"""BioCLIP text embeddings for every iWildCam class (row y of categories.csv -> row y of the output).
Prompts are fixed a priori (not tuned): "a photo of <scientific name>." and, for class 0 (empty),
"a camera trap photo with no animal."."""
import numpy as np
import open_clip
import pandas as pd
import torch

ROOT = "${FEDPCT_ROOT}"
cats = pd.read_csv(f"{ROOT}/data/iwildcam_v2.0/categories.csv")
cats = cats[cats.y != 99999].sort_values("y")  # 99999 = categories absent from the WILDS data
assert (cats.y.to_numpy() == np.arange(len(cats))).all()
assert set(pd.read_csv(f"{ROOT}/features/iwildcam_meta.csv").y) <= set(cats.y)
prompts = ["a camera trap photo with no animal." if n == "empty" else f"a photo of {n}." for n in cats.name]

model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
model = model.cuda().eval()
with torch.no_grad():
    T = model.encode_text(tok(prompts).cuda()).float()
T = torch.nn.functional.normalize(T, dim=-1).cpu().numpy()
np.save(f"{ROOT}/features/iwildcam_text_bioclip.npy", T)
print("saved", T.shape, "| logit_scale", float(model.logit_scale.exp()))
