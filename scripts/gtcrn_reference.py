# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "soundfile", "einops"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Run the upstream PyTorch GTCRN (offline) reference on a WAV file.

Writes the enhanced waveform so you can A/B compare it against the svod
streaming / offline examples.

Usage:
  uv run scripts/gtcrn_reference.py --in audio_clip.wav --out /tmp/enhanced_ref.wav
  uv run scripts/gtcrn_reference.py --in audio_clip.wav --out /tmp/enhanced_ref.wav \
      --checkpoint submodules/gtcrn/checkpoints/model_trained_on_dns3.tar

The input must be 16 kHz mono. The script STFTs with n_fft=512, hop=256,
hann_window(512)**0.5 (matching `torch.stft` in `infer.py`), runs the upstream
`GTCRN.forward`, and ISTFTs the enhanced spectrogram back to PCM.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Upstream constants (gtcrn.py / infer.py).
NFFT = 512
HOP = 256


def load_gtcrn_module():
    """Import the upstream `gtcrn.py` from the submodule."""
    p = Path(__file__).resolve().parents[1] / "submodules/gtcrn/gtcrn.py"
    spec = importlib.util.spec_from_file_location("gtcrn", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="input WAV (16 kHz mono)")
    ap.add_argument("--out", type=Path, required=True, help="output enhanced WAV")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("submodules/gtcrn/checkpoints/model_trained_on_dns3.tar"),
        help="upstream .tar checkpoint",
    )
    args = ap.parse_args()

    samples, fs = sf.read(str(args.inp), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]  # mono: take first channel
    assert fs == 16000, f"expected 16 kHz, got {fs}"
    print(f"Input: {args.inp} ({len(samples)} samples, {len(samples)/fs:.1f}s)")

    GTCRN = load_gtcrn_module().GTCRN
    pkg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GTCRN().eval()
    model.load_state_dict(pkg["model"], strict=True)

    win = torch.hann_window(NFFT).pow(0.5)
    x = torch.from_numpy(samples)[None]  # [1, L]
    spec = torch.stft(x, NFFT, HOP, NFFT, win, return_complex=False)  # [1, F, T, 2]
    print(f"STFT: {tuple(spec.shape)}")

    with torch.no_grad():
        enhanced_spec = model(spec)  # [1, F, T, 2]
    # istft needs a complex tensor on modern PyTorch; GTCRN returns [..., 2].
    enh_cplx = torch.view_as_complex(enhanced_spec[0].contiguous())  # [F, T]
    enh = torch.istft(enh_cplx, NFFT, HOP, NFFT, win, return_complex=False)  # [L']

    sf.write(str(args.out), enh.detach().cpu().numpy().astype(np.float32), fs)
    print(f"Wrote {args.out} ({len(enh)} samples)")


if __name__ == "__main__":
    main()
