from typing import Dict, Optional, Sequence

import torch

from .models.bands import BandSplit, fft2c, r2c, radius_map


def crop_center(x: torch.Tensor, size=(320, 320)) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    th, tw = min(size[0], h), min(size[1], w)
    i, j = (h - th) // 2, (w - tw) // 2
    return x[..., i:i + th, j:j + tw]


def magnitude(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(dim=1).sqrt()


def psnr(pred: torch.Tensor, gt: torch.Tensor,
         peak: Optional[torch.Tensor] = None, eps: float = 1e-12):
    if peak is None:
        peak = gt.amax(dim=(-2, -1))
    mse = (pred - gt).pow(2).mean(dim=(-2, -1))
    return 20 * torch.log10(peak.clamp_min(eps)
                            / mse.clamp_min(eps).sqrt())
