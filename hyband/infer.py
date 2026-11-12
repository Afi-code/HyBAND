import argparse
import csv
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(REPO.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastmri.evaluate import psnr, ssim, nmse

from data import Config, build_loader
from hyband.trainer import build_model

def score_crop(t, size=(320, 320)):
    h, w = t.shape[-2], t.shape[-1]
    ch, cw = min(size[0], h), min(size[1], w)
    top, left = (h - ch) // 2, (w - cw) // 2
    return t[..., top:top + ch, left:left + cw]


def mag(x):
    return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)

if __name__ == "__main__":
    main()
