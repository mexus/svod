#!/usr/bin/env python3
"""Generate the PyTorch fixtures for the YOLO26n parity test.

Writes to data/yolo/:
  - model.safetensors    the checkpoint weights, fp32, PyTorch key names
  - golden.safetensors
      images        [1, 3, 640, 640] f32 -- the deterministic test image
      images_shape  [4] i64
      output        [1, 84, 8400] f32 -- what Yolo26Detect::forward returns

Usage:
  pip install ultralytics safetensors torch
  python scripts/convert_yolo.py  # writes both files to data/yolo/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ASSET = "yolo26n.pt"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo"
SIDE = 640


def deterministic_image() -> torch.Tensor:
    """The gradient pattern the Rust example and the parity test both build."""
    c = np.arange(3, dtype=np.int64)[:, None, None]
    h = np.arange(SIDE, dtype=np.int64)[None, :, None]
    w = np.arange(SIDE, dtype=np.int64)[None, None, :]
    img = (c * 213 + h + w).astype(np.float32) / np.float32(SIDE + SIDE + 3 * 213)
    return torch.from_numpy(img[None])


def raw_predictions(net: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The decoded `[B, 4 + nc, A]` tensor, before top-k selection.

    YOLO26's head is end-to-end, so calling the model returns
    `(postprocessed, aux)` where `postprocessed` is `[B, max_det, 6]` -- boxes
    already selected and paired with a score and a class. `Yolo26Detect::forward`
    stops earlier than that, so comparing against it means re-running the head's
    own inference step over the branch outputs `aux` carries.
    """
    with torch.no_grad():
        out = net(images)
    if not (isinstance(out, tuple) and isinstance(out[1], dict) and "one2one" in out[1]):
        raise RuntimeError(
            f"Expected an end-to-end head returning (preds, {{'one2one': ...}}), got {type(out)}. "
            "Is this checkpoint really a YOLO26?"
        )
    detect = net.model[-1]
    with torch.no_grad():
        return detect._inference(out[1]["one2one"])


def main() -> None:
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Give Ultralytics an explicit path, otherwise it downloads the asset
    # relative to the working directory and litters the repository root.
    net = YOLO(str(OUTPUT_DIR / ASSET)).model.float().eval()

    weights = {k: v.contiguous() for k, v in net.state_dict().items() if v.is_floating_point()}
    save_file(weights, str(OUTPUT_DIR / "model.safetensors"))
    print(f"saved {len(weights)} tensors -> {OUTPUT_DIR / 'model.safetensors'}")

    images = deterministic_image()
    output = raw_predictions(net, images)
    print(f"output shape: {tuple(output.shape)}")

    save_file(
        {
            "images": images,
            "images_shape": torch.tensor(list(images.shape), dtype=torch.int64),
            "output": output.contiguous(),
        },
        str(OUTPUT_DIR / "golden.safetensors"),
    )
    print(f"saved {OUTPUT_DIR / 'golden.safetensors'}")


if __name__ == "__main__":
    main()
