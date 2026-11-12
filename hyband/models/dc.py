import torch
import torch.nn as nn
import torch.nn.functional as F

from models.data_consistency import Encoder, fft2c, ifft2c, zdot


def _inv_softplus(v: float) -> float:
    v = max(float(v), 1e-6)
    return v + float(torch.log(-torch.expm1(torch.tensor(-v))))

def mu_map(mus, windows):
    return mus[0] * windows[0] + mus[1] * windows[1] + mus[2] * windows[2]


def hard_dc(x, sens_maps, y, mask, w_lf):
    k = fft2c(sens_maps * x[:, None])
    m = (mask * w_lf)[:, None].to(k.real.dtype)
    d = torch.sum(ifft2c(m * (y - k)) * torch.conj(sens_maps), dim=1)
    return x + ifft2c(fft2c(d) * w_lf.to(k.real.dtype))

hard_lf = hard_dc
