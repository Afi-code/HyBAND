import math

import numpy as np
import torch

K7 = torch.ones(1, 1, 7, 7) / 49.0


def psnr(gt, pred, data_range=None):
    mse = float(((pred - gt) ** 2).mean())
    rng = float(gt.max()) if data_range is None else float(data_range)
    return 20 * math.log10(max(rng, 1e-30) / math.sqrt(max(mse, 1e-30)))


def ssim_map(gt, pred, data_range=None):
    mx = float(gt.max()) if data_range is None else float(data_range)
    c1, c2 = (0.01 * mx) ** 2, (0.03 * mx) ** 2
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a),
                                  dtype=torch.float32)[None, None]
    f = lambda a: torch.nn.functional.conv2d(a, K7, padding=3)[0, 0]
    G, P = t(gt), t(pred)
    m1, m2 = f(G), f(P)
    v1 = f(G * G) - m1 * m1
    v2 = f(P * P) - m2 * m2
    cv = f(G * P) - m1 * m2
    return (((2*m1*m2 + c1) * (2*cv + c2))
            / ((m1**2 + m2**2 + c1) * (v1 + v2 + c2))).numpy()


def ssim(gt, pred, data_range=None):
    return float(ssim_map(gt, pred, data_range).mean())
