"""Lazy LPIPS backend (torch). Imported only when the ``lpips`` metric is used.

Kept isolated from ``metrics.py`` so importing the metric registry never pulls
in torch. The LPIPS network is loaded once and cached on the module.
"""
from __future__ import annotations

import numpy as np

_net = None  # cached lpips.LPIPS network


def _get_net():
    global _net
    if _net is None:
        import lpips  # heavy import
        import torch

        net = lpips.LPIPS(net="alex")
        net.eval()
        if torch.cuda.is_available():
            net = net.to("cuda")
        _net = net
    return _net


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


def lpips_video_distance(gen: np.ndarray, truth: np.ndarray) -> float:
    """Mean per-frame LPIPS distance over the shortest common frame count."""
    import torch

    net = _get_net()
    g = _to_tensor(gen)
    t = _to_tensor(truth)
    n = min(g.shape[0], t.shape[0])
    if n == 0:
        raise ValueError("no frames to compare")
    g, t = g[:n], t[:n]
    device = next(net.parameters()).device
    g, t = g.to(device), t.to(device)
    with torch.no_grad():
        d = net(g, t)  # (n, 1, 1, 1)
    return float(d.mean().item())
