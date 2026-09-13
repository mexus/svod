# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "soundfile", "safetensors", "huggingface_hub", "einops"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Convert a GTCRN checkpoint (.tar) to the svod safetensors layout and emit a
PyTorch golden for the Rust parity tests.

Usage:
  uv run scripts/convert_gtcrn.py \\
      --checkpoint submodules/gtcrn/checkpoints/model_trained_on_dns3.tar \\
      [--golden submodules/gtcrn/test_wavs/mix.wav] [--selfcheck]

The script does two jobs:

convert
    Load ``ckpt["model"]`` from the upstream ``.tar`` and rewrite it into svod's
    state-dict convention. The only non-trivial remap is the GRU gate order:
    PyTorch's ``nn.GRU`` stores gate rows as ``[reset, update, new]`` (``r, z,
    n``), while svod's ``gru()`` op expects ``[z, r, h]``. For every
    ``*_l0`` / ``*_l0_reverse`` weight and bias tensor we swap the first two
    hidden-sized gate blocks (rows ``0..H`` <-> ``H..2H``) and leave the
    candidate-gate block (``2H..3H``) in place. ``num_batches_tracked`` entries
    are dropped; ``running_var`` keys are kept verbatim, which is what
    ``svod_tensor::nn::BatchNorm2d`` consumes directly (with ``eps = 1e-5``,
    PyTorch's default) -- there is no fold step on either side.

    The Rust side must call svod's ``gru()`` with ``linear_before_reset(1)``:
    PyTorch's ``nn.GRU`` computes the candidate gate as
    ``tanh(x @ Wn + r * (h @ Rn + rb_n) + wb_n)`` (the reset gate multiplies the
    *projected* hidden state), which is svod's ``linear_before_reset=1`` path —
    the default ``=0`` path does not match PyTorch.

    The bidirectional DPGRNN intra-RNNs expose two GRUs each (``rnn1``, ``rnn2``)
    with forward (``_l0``) and reverse (``_l0_reverse``) weight sets. They are
    emitted as ``<path>.rnn1.weight_ih_l0_f`` / ``..._b`` (and ``_i_h`` /
    ``_h_h`` short forms) — see the remap table below.

golden
    Run the upstream ``GTCRN`` (imported from ``submodules/gtcrn/gtcrn.py``) on
    the bundled ``mix.wav`` and store ``samples`` (f32 [-1,1]), ``spec``
    (the ``[1, 257, T, 2]`` complex spectrogram fed to ``forward``), ``output``
    (the ``[1, 257, T, 2]`` enhanced spectrogram) and ``enh`` (the inverse-STFT
    waveform) for the Rust parity test.

selfcheck
    Reproduce svod's GRU gate equations (``[z, r, h]`` order,
    ``linear_before_reset=1``, which is what ``nn.GRU`` implements and what the
    Rust side sets) in NumPy on a tiny GRU with the permuted weights, and assert
    agreement with PyTorch to 1e-6. Guards the gate-row permutation in isolation
    before the Rust side is involved.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file

# Upstream GTCRN constants (gtcrn.py: GTCRN.__init__).
NFFT = 512
HOP = 256
ERB_SUBBAND_1 = 65
ERB_SUBBAND_2 = 64


# --------------------------------------------------------------------------- #
# GRU gate remap
# --------------------------------------------------------------------------- #

def permute_gru_gates(t: torch.Tensor, hidden: int) -> torch.Tensor:
    """Reorder gate rows from PyTorch ``[r, z, n]`` to svod ``[z, r, h]``.

    ``t`` is a leading-``3*hidden`` gate tensor: ``weight_ih``/``weight_hh``
    are ``[3*hidden, *]`` and the four ``bias_*`` are ``[3*hidden]``. The first
    two hidden-sized blocks are swapped; the third (candidate/new) stays put.
    """
    assert t.shape[0] == 3 * hidden, f"expected leading 3*hidden={3 * hidden}, got {t.shape[0]}"
    r, z, n = t[:hidden], t[hidden : 2 * hidden], t[2 * hidden :]
    return torch.cat([z, r, n], dim=0).contiguous()


def gru_hidden_for(prefix: str, sd: dict) -> int:
    """Infer the GRU hidden size for a ``<prefix>.weight_ih_l0`` key."""
    return sd[f"{prefix}.weight_ih_l0"].shape[0] // 3


def remap(sd: dict) -> dict:
    """Apply the GRU gate permutation + bidirectional key split; return the
    svod-layout state dict.

    Two transforms:

    1. **Gate permutation.** Every ``*_l0`` / ``*_l0_reverse`` GRU weight and
       bias tensor has its first two hidden-sized gate blocks swapped
       (PyTorch ``[r, z, n]`` → svod ``[z, r, h]``).

    2. **Bidirectional key split.** The DPGRNN's bidirectional GRNNs expose
       ``rnn1``/``rnn2`` modules whose forward weights are
       ``<prefix>.rnn1.weight_ih_l0`` and reverse weights are
       ``<prefix>.rnn1.weight_ih_l0_reverse``. Svod stores the two directions
       as separate ``GruWeights`` under ``rnn1_f`` / ``rnn1_b``, so the reverse
       keys are renamed ``..._l0_reverse`` → ``rnn1_b.weight_ih_l0`` (etc.) and
       the forward keys ``rnn1.`` → ``rnn1_f.``. Unidirectional GRNNs and the
       TRA att_gru keep their ``rnn1.`` / ``att_gru.`` prefixes (no ``_b``).

    BN ``running_var`` keys pass through unchanged (svod folds them at load
    time); ``num_batches_tracked`` entries are dropped.
    """
    suffixes = (
        "weight_ih_l0_reverse", "weight_hh_l0_reverse", "bias_ih_l0_reverse", "bias_hh_l0_reverse",
        "weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
    )

    out = {}
    for key, val in sd.items():
        if key.endswith("num_batches_tracked"):
            continue

        matched = next((s for s in suffixes if key.endswith(s)), None)
        if matched is None:
            out[key] = val
            continue

        # Gate permutation (same for forward and reverse).
        prefix = key[: -(len(matched) + 1)]
        hidden = gru_hidden_for(prefix, sd)
        val = permute_gru_gates(val, hidden)

        # Bidirectional key split: <prefix>.rnn1.<suffix>_reverse ->
        # <prefix>.rnn1_b.<bare_suffix>; <prefix>.rnn1.<suffix> ->
        # <prefix>.rnn1_f.<bare_suffix>. Only rnn1/rnn2 under intra_rnn/inter_rnn
        # split this way; att_gru keeps its name and forward-only keys stay.
        bare = matched.replace("_reverse", "")
        if key.endswith("_reverse"):
            # Rename "...rnn1.weight_ih_l0_reverse" -> "...rnn1_b.weight_ih_l0".
            stem = key[: -len("_reverse")]
            new_key = stem.replace(".rnn1.", ".rnn1_b.").replace(".rnn2.", ".rnn2_b.")
        else:
            new_key = key.replace(".rnn1.", ".rnn1_f.").replace(".rnn2.", ".rnn2_f.")
        out[new_key] = val

    return out


# --------------------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------------------- #

def convert(checkpoint: Path, out: Path) -> dict:
    pkg = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = pkg["model"]
    remapped = {k: v.contiguous().float() for k, v in remap(sd).items()}

    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(remapped, str(out))
    n_params = sum(v.numel() for v in remapped.values() if v.dtype.is_floating_point)
    print(f"wrote {out} ({len(remapped)} tensors, {n_params} model params)")
    return sd


# --------------------------------------------------------------------------- #
# Selfcheck: svod GRU equations vs PyTorch, on permuted weights
# --------------------------------------------------------------------------- #

def selfcheck() -> None:
    """Verify the [r,z,n] -> [z,r,h] gate permutation against svod's GRU math.

    Builds a tiny nn.GRU, permutes its weights into svod layout, then runs svod's
    documented recurrence (rnn.rs:151-159) with ``linear_before_reset=1`` — the
    formulation PyTorch's ``nn.GRU`` actually uses:

        z = sigmoid(X @ Wz^T + H @ Rz^T + wb_z + rb_z)
        r = sigmoid(X @ Wr^T + H @ Rr^T + wb_r + rb_r)
        h = tanh(X @ Wh^T + r * (H @ Rh^T + rb_h) + wb_h)
        H_new = (1 - z) * h + z * H_prev

    and compares the full hidden-state sequence to PyTorch's GRU output. The
    ``linear_before_reset=1`` flag must be passed to svod's ``gru()`` op for
    every GTCRN GRU; the default ``=0`` path does not match PyTorch.
    """
    torch.manual_seed(0)
    seq, batch, inp, hidden = 5, 2, 3, 4
    ref = torch.nn.GRU(inp, hidden, 1, batch_first=True).eval()
    x = torch.randn(batch, seq, inp)

    with torch.no_grad():
        want, _ = ref(x)  # [batch, seq, hidden]

        # Pull + permute into svod [z, r, h] order.
        w_ih = permute_gru_gates(ref.weight_ih_l0, hidden)  # [3H, inp]
        w_hh = permute_gru_gates(ref.weight_hh_l0, hidden)  # [3H, H]
        b_ih = permute_gru_gates(ref.bias_ih_l0, hidden)    # [3H]
        b_hh = permute_gru_gates(ref.bias_hh_l0, hidden)    # [3H]

        def gate(w, b, name):
            h_size = w.shape[0] // 3
            i = {"z": 0, "r": 1, "h": 2}[name] * h_size
            return w[i : i + h_size], b[i : i + h_size]

        wz, bz = gate(w_ih, b_ih, "z")
        wr, br = gate(w_ih, b_ih, "r")
        wh, bh = gate(w_ih, b_ih, "h")
        rz, rbz = gate(w_hh, b_hh, "z")
        rr, rbr = gate(w_hh, b_hh, "r")
        rh, rbh = gate(w_hh, b_hh, "h")

        h = torch.zeros(batch, hidden)
        outs = []
        for t in range(seq):
            xt = x[:, t]
            z = torch.sigmoid(xt @ wz.T + h @ rz.T + bz + rbz)
            r = torch.sigmoid(xt @ wr.T + h @ rr.T + br + rbr)
            # linear_before_reset=1: r multiplies (H @ Rh^T + rb_h), not (r*H) @ Rh^T.
            cand = torch.tanh(xt @ wh.T + r * (h @ rh.T + rbh) + bh)
            h = (1 - z) * cand + z * h
            outs.append(h)
        got = torch.stack(outs, dim=1)  # [batch, seq, hidden]

    err = (want - got).abs().max().item()
    assert err < 1e-6, f"GRU gate permutation mismatch: max |delta| = {err}"
    print(f"selfcheck OK: permuted svod GRU matches PyTorch (max |delta| = {err:.2e})")


# --------------------------------------------------------------------------- #
# Golden
# --------------------------------------------------------------------------- #

def load_gtcrn_module():
    """Import submodules/gtcrn/gtcrn.py by path."""
    path = Path(__file__).resolve().parents[1] / "submodules/gtcrn/gtcrn.py"
    spec = importlib.util.spec_from_file_location("gtcrn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_wav_mono_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    return np.ascontiguousarray(data, dtype=np.float32)


def golden_stream(checkpoint: Path, wav: Path, out: Path) -> None:
    """Generate a golden from the upstream ``StreamGTCRN`` (frame-by-frame).

    This is the correct parity target for the Rust streaming model — the
    streaming architecture (StreamConvTranspose2d) is causally different from
    the offline model, so the offline golden does not apply.
    """
    import importlib.util as ilu
    stream_path = Path(__file__).resolve().parents[1] / "submodules/gtcrn/stream/gtcrn_stream.py"
    spec_mod = ilu.spec_from_file_location("gtcrn_stream", stream_path)
    stream_module = ilu.module_from_spec(spec_mod)
    # The stream module imports from ``modules``, so we need its parent on sys.path.
    import sys
    stream_dir = str(stream_path.parent)
    modules_dir = str(stream_path.parent / "modules")
    for d in (stream_dir, modules_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec_mod.loader.exec_module(stream_module)
    StreamGTCRN = stream_module.StreamGTCRN
    # convert_to_stream is in modules/convert.py (now on sys.path).
    from convert import convert_to_stream

    samples = read_wav_mono_16k(wav)
    pkg = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Build offline + stream models, convert weights.
    # The stream module's gtcrn.py is the offline model used as the source.
    from gtcrn import GTCRN
    offline = GTCRN().eval()
    offline.load_state_dict(pkg["model"], strict=True)
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline)

    win = torch.hann_window(NFFT).pow(0.5)
    x = torch.from_numpy(samples)[None]
    spec = torch.stft(x, NFFT, HOP, NFFT, win, return_complex=False)

    parity_frames = 24
    spec_crop = spec[:, :, :parity_frames, :]  # [1, F, 24, 2]

    # Run the streaming model frame-by-frame.
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)
    outs = []
    with torch.no_grad():
        for i in range(parity_frames):
            xi = spec_crop[:, :, i:i+1, :]  # [1, F, 1, 2]
            yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
            outs.append(yi)
    stream_out = torch.cat(outs, dim=2)[0]  # [F, 24, 2]

    out.parent.mkdir(parents=True, exist_ok=True)
    save_np(
        {
            "samples": samples,
            "spec_crop": spec_crop.contiguous().numpy().astype(np.float32),
            "stream_output_crop": stream_out[None].contiguous().numpy().astype(np.float32),
        },
        str(out),
    )
    print(f"wrote {out}: stream_output_crop {stream_out[None].shape}")


def golden(checkpoint: Path, wav: Path, out: Path) -> None:
    GTCRN = load_gtcrn_module().GTCRN

    samples = read_wav_mono_16k(wav)
    pkg = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GTCRN().eval()
    model.load_state_dict(pkg["model"], strict=True)

    win = torch.hann_window(NFFT).pow(0.5)
    x = torch.from_numpy(samples)[None]  # [1, L]
    spec = torch.stft(x, NFFT, HOP, NFFT, win, return_complex=False)  # [1, F, T, 2]
    with torch.no_grad():
        enhanced_spec = model(spec)[0]  # [F, T, 2]
        # Cropped parity reference: forward on the first PARITY_FRAMES time
        # steps. The Rust test crops the input identically, so this is the exact
        # comparison target (the GRU recurrence makes a slice of the full-length
        # output an invalid target).
        parity_frames = 24
        spec_crop = spec[:, :, :parity_frames, :]
        out_crop = model(spec_crop)[0]  # [F, parity_frames, 2]
    # istft needs a complex tensor on modern PyTorch; GTCRN's forward returns
    # a [..., 2] real/imag layout, so view it as complex for the inverse.
    enh_cplx = torch.view_as_complex(enhanced_spec.contiguous())
    enh = torch.istft(enh_cplx, NFFT, HOP, NFFT, win, return_complex=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_np(
        {
            "samples": samples,
            # .contiguous() is load-bearing: torch.stft(return_complex=False) and
            # the model output are both non-contiguous, and safetensors writes
            # MEMORY order under the logical-shape header -- without it `spec`
            # lands as [T, F, 2] and `output` as [2, T, F] while both claim
            # [1, 257, T, 2], so `spec_crop != spec[:, :, :24]` in the same file.
            "spec": spec.contiguous().numpy().astype(np.float32),      # [1, 257, T, 2]
            "spec_shape": np.array(spec.shape, dtype=np.int64),      # [1, 257, T, 2]
            "output": enhanced_spec[None].contiguous().numpy().astype(np.float32),  # [1, 257, T, 2]
            "enh": enh.numpy().astype(np.float32),                    # [L']
            "spec_crop": spec_crop.contiguous().numpy().astype(np.float32),       # [1, 257, 24, 2]
            "output_crop": out_crop[None].contiguous().numpy().astype(np.float32),  # [1, 257, 24, 2]
        },
        str(out),
    )
    print(f"wrote {out}: samples {samples.shape}, spec {tuple(spec.shape)}, enh {tuple(enh.shape)}")
    print(f"spec[:2,0,0] = {spec[0, :2, 0, 0].tolist()}; output[:2,0,0] = {enhanced_spec[:2, 0, 0].tolist()}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("submodules/gtcrn/checkpoints/model_trained_on_dns3.tar"))
    ap.add_argument("--out", type=Path, default=Path("data/gtcrn/gtcrn.safetensors"))
    ap.add_argument("--golden", type=Path, help="wav file for golden generation")
    ap.add_argument("--golden-out", type=Path, default=Path("data/gtcrn/golden.safetensors"))
    ap.add_argument("--selfcheck", action="store_true", help="verify the GRU gate permutation")
    ap.add_argument("--stream-golden", type=Path, help="wav file for streaming golden generation")
    ap.add_argument("--stream-golden-out", type=Path, default=Path("data/gtcrn/golden_stream.safetensors"))
    args = ap.parse_args()

    sd = convert(args.checkpoint, args.out)
    if args.selfcheck:
        selfcheck()
    if args.golden:
        golden(args.checkpoint, args.golden, args.golden_out)
    if args.stream_golden:
        golden_stream(args.checkpoint, args.stream_golden, args.stream_golden_out)


if __name__ == "__main__":
    main()
