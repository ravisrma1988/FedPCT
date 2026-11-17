"""Split an open_clip VisionTransformer into a frozen trunk (embeddings + blocks[:CUT]) and a
trainable top (blocks[CUT:] + pooling + projection), plus a DALI eval pipeline shared by the
caching / fine-tuning / extraction scripts."""
import copy

import open_clip
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

MODEL = "hf-hub:imageomics/bioclip-2"
CUT = 22  # blocks 22, 23 are trainable


def load():
    model, _, _ = open_clip.create_model_and_transforms(MODEL)
    v = model.visual.cuda().eval()
    cfg = getattr(v, "preprocess_cfg", {}) or {}
    size = cfg.get("size", v.image_size)
    size = size[0] if isinstance(size, (tuple, list)) else size
    return v, size, cfg.get("mean", open_clip.OPENAI_DATASET_MEAN), cfg.get("std", open_clip.OPENAI_DATASET_STD)


def trunk(v, img):
    x = v._embeds(img)
    for r in v.transformer.resblocks[:CUT]:
        x = r(x)
    return x


class Top(torch.nn.Module):
    """Trainable top of the tower; forward(tokens) -> unit-norm embedding."""

    def __init__(self, v):
        super().__init__()
        self.blocks = copy.deepcopy(v.transformer.resblocks[CUT:])
        # Keep the frozen tower OUT of the module registry (no params/state_dict of 300M weights);
        # it is only used for its _pool logic. ln_post / proj are copies, so training never edits v.
        object.__setattr__(self, "_v", v)
        self.ln_post = copy.deepcopy(v.ln_post)
        self.proj = torch.nn.Parameter(v.proj.detach().clone())

    def forward(self, x):
        for r in self.blocks:
            x = r(x)
        v = self._v
        ln, v.ln_post = v.ln_post, self.ln_post
        try:
            pooled, _ = v._pool(x)
        finally:
            v.ln_post = ln
        return torch.nn.functional.normalize(pooled @ self.proj, dim=-1)


def dali_iter(files, size, mean, std, batch, threads=16):
    @pipeline_def(batch_size=batch, num_threads=threads, device_id=0, prefetch_queue_depth=4)
    def pipe():
        jpegs, idx = fn.readers.file(files=files, labels=list(range(len(files))), random_shuffle=False,
                                     name="reader")
        img = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        img = fn.resize(img, resize_shorter=size, interp_type=types.INTERP_CUBIC, antialias=True)
        img = fn.crop_mirror_normalize(img, crop=(size, size), dtype=types.FLOAT,
                                       mean=[255 * m for m in mean], std=[255 * s for s in std],
                                       output_layout="CHW")
        return img, idx

    p = pipe()
    p.build()
    return DALIGenericIterator(p, ["img", "idx"], reader_name="reader",
                               last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=False)
