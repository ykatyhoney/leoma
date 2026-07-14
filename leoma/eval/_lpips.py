"""Lazy LPIPS backend (torch). Imported only when the ``lpips`` metric is used.

Kept isolated from ``metrics.py`` so importing the metric registry never pulls in
torch. The LPIPS network is loaded once per device and cached on the module.

**Scoring runs on CPU in production** (``duel.metric_device = "cpu"``). Generation
on a 14B bf16 diffusion model is already the one step that cannot be made
bit-exact across GPU architectures; scoring on the GPU as well would add a *second*
source of cross-validator noise for no benefit. On CPU in fp32 the metric is an
exactly reproducible function of the frames, so any distance disagreement between
two validators is attributable to generation alone. It costs minutes against the
*hours* the generations take.
"""
from __future__ import annotations

import numpy as np

_nets: dict[str, object] = {}  # device -> cached lpips.LPIPS network

DEFAULT_DEVICE = "cpu"


def _get_net(device: str):
    net = _nets.get(device)
    if net is None:
        import lpips  # heavy import
        import torch

        net = lpips.LPIPS(net="alex")
        net.eval()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("metric_device='cuda' but no CUDA device is available")
        net = net.to(device)
        _nets[device] = net
    return net


def _to_tensor(frames: np.ndarray):
    """(T, H, W, C) uint8/float -> torch (T, C, H, W) normalized to [-1, 1]."""
    import torch

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 3:  # (T, H, W) grayscale -> add channel, expand to 3
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.max() > 1.5:  # 8-bit range -> [0, 1]
        arr = arr / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    return t * 2.0 - 1.0  # [0,1] -> [-1,1]


def lpips_video_distance(gen: np.ndarray, truth: np.ndarray, *, device: str = DEFAULT_DEVICE) -> float:
    """Mean per-frame LPIPS distance. The truth's length is authoritative."""
    import torch

    g = _to_tensor(gen)
    t = _to_tensor(truth)
    n = t.shape[0]
    if n == 0:
        raise ValueError("no frames to compare")
    if g.shape[0] < n:
        # metrics.require_frames() already rejects this; belt and braces, because
        # silently scoring a short generation against a truncated truth is exactly
        # the freeze cheat.
        raise ValueError(f"generation too short: {g.shape[0]} frames < {n} ground-truth frames")
    g, t = g[:n], t[:n]

    net = _get_net(device)
    g, t = g.to(device), t.to(device)
    with torch.no_grad():
        d = net(g, t)  # (n, 1, 1, 1)
    return float(d.mean().item())
