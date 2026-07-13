"""Torch determinism, applied honestly.

What these flags buy: **run-to-run reproducibility on one box**. Run the same duel
twice on the same GPU and you get the same frames, so a disputed verdict can
actually be investigated.

What they do **not** buy, and no flag will: bit-exactness across GPU architectures
for a 14B bf16 diffusion model. Kernel selection, reduction order and tensor-core
paths differ between an H100 and an A100, and the difference compounds over 30
denoising steps.

So the subnet does not pretend generation is exact. It does three things instead:

1. **Removes every *other* source of noise** — the corpus is hash-pinned, the
   scoring runs on CPU in fp32 (``metric_device="cpu"``), and every parameter is
   pinned and echoed. Generation is left as the *only* fuzzy step.
2. **Makes the residual noise visible** — the verdict carries per-clip distances
   *and* per-clip generated-frame digests, so validators can diff at three levels:
   frames (bit-exact?), distances (close?), verdict (same crown?).
3. **Sets ``delta_threshold`` above the noise** — a challenger must win by a margin
   wider than cross-GPU jitter. Calibrating that margin against a real fleet is the
   single largest open consensus risk in the subnet, and it is a measurement, not a
   code change.

:func:`runtime_env` records what the box actually was, so a divergence can be
attributed rather than argued about.
"""
from __future__ import annotations

import os
import platform
import sys

# CuBLAS needs this set *before* the CUDA context is created; setting it after
# torch has already initialized is a silent no-op, which is why it lives here and
# not inside apply_determinism().
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def apply_determinism(spec) -> None:
    """Apply the pinned :class:`~leoma.eval.spec.DeterminismSpec` to torch."""
    if spec.torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

    import torch

    torch.backends.cudnn.benchmark = spec.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = spec.allow_tf32
    torch.backends.cudnn.allow_tf32 = spec.allow_tf32

    if spec.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        # warn_only: a handful of diffusers ops have no deterministic kernel. Hard
        # -failing there would take the subnet down for a marginal gain, and the
        # verdict's frame digests would surface the nondeterminism anyway.
        torch.use_deterministic_algorithms(True, warn_only=True)


def runtime_env() -> dict:
    """What this box is — recorded in the verdict so divergence is attributable."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "gpu": None,
        "cuda": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — a missing torch must never break a verdict
        pass
    return env


__all__ = ["CUBLAS_WORKSPACE_CONFIG", "apply_determinism", "runtime_env"]
