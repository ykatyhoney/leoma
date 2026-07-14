"""Lazy CLIP backend (open_clip). Imported only for the ``clip`` metric.

Semantic-fidelity distance: embed the generation frames and the real-continuation
frames with a CLIP image encoder and take the mean per-frame cosine distance.
Unlike pixel/perceptual metrics, this is forgiving of *plausible* divergence —
it asks "does this look like the same scene/content", not "do the pixels match" —
which helps with the one-to-many nature of video generation.

``torch``/``open_clip`` are imported lazily; the cosine-distance math
(``cosine_distance``) is pure numpy and unit-testable. Like LPIPS this runs on
GPU, so it carries the same cross-validator determinism caveat.
"""
from __future__ import annotations

import numpy as np

_models: dict[str, tuple] = {}  # device -> (model, preprocess)

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

DEFAULT_DEVICE = "cpu"


def _get_model(device: str):
    cached = _models.get(device)
    if cached is None:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        model.eval()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("metric_device='cuda' but no CUDA device is available")
        model = model.to(device)
        cached = (model, preprocess)
        _models[device] = cached
    return cached


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine distance (1 - cos) between two (N, D) embedding sets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float(np.mean(1.0 - (a * b).sum(axis=-1)))


def _embed(frames: np.ndarray, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    model, preprocess = _get_model(device)
    batch = torch.stack([
        preprocess(Image.fromarray(np.asarray(f).astype("uint8")))
        for f in frames
    ]).to(device)
    with torch.no_grad():
        emb = model.encode_image(batch)
    return emb.detach().cpu().numpy()


def clip_video_distance(gen: np.ndarray, truth: np.ndarray, *, device: str = DEFAULT_DEVICE) -> float:
    """Mean per-frame CLIP cosine distance between generation and truth."""
    g = np.asarray(gen)
    t = np.asarray(truth)
    if g.shape[0] < t.shape[0]:
        raise ValueError(
            f"generation too short: {g.shape[0]} frames < {t.shape[0]} ground-truth frames"
        )
    n = t.shape[0]
    if n == 0:
        # Used to return 0.0 — a PERFECT score for an empty generation.
        raise ValueError("no frames to compare")
    return cosine_distance(_embed(g[:n], device), _embed(t[:n], device))
