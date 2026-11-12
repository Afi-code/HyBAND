import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from . import registry as reg

for _p in (str(reg.HYB), str(reg.PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from data import Config, build_loader

FLIP = lambda a: np.ascontiguousarray(a[..., ::-1, :])


def crop(a, n=reg.CROP):
    hh, ww = a.shape[-2], a.shape[-1]
    i, j = (hh - n) // 2, (ww - n) // 2
    return a[..., i:i + n, j:j + n]


def annotations():
    meta = {d.name: json.load(open(d / "metadata.json"))
            for d in sorted(reg.ROOT.iterdir())
            if (d / "metadata.json").is_file()}
    out = defaultdict(list)
    for r in csv.DictReader(open(reg.ANNOTATIONS)):
        m = meta.get(r["file"])
        if m is None or r["x"] in ("", None):
            continue
        s = int(r["slice"])
        if not (m["slice_start"] <= s <= m["slice_end_inclusive"]):
            continue
        try:
            x, y, w, h = (int(r["x"]), int(r["y"]),
                          int(r["width"]), int(r["height"]))
        except ValueError:
            continue
        out[(r["file"] + ".h5", s - m["slice_start"])].append(
            (x, y, w, h, r["label"].strip()))
    return out


def largest_per_label(ann):
    best = {}
    for (vol, sl), lst in ann.items():
        for (x, y, w, h, lab) in lst:
            if lab not in best or w * h > best[lab][0]:
                best[lab] = (w * h, vol, sl, x, y, w, h, lab)
    return sorted(best.values(), key=lambda p: p[7])


def loader(split="test", num_workers=4, device=None):
    cfg = Config(data_path=reg.DATA, dataset=reg.DATASET,
                 accelerations=[reg.ACCEL], center_fractions=[],
                 acs_lines=reg.ACS_LINES, mask_type="equispaced",
                 batch_size=1, num_workers=num_workers, smaps_cache=None,
                 use_dataset_cache=False)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_loader(cfg, split, shuffle=False, device=dev, ssdu_split=False)


def target_of(batch):
    gt = batch.target.float()
    mag = torch.sqrt(gt[0, 0] ** 2 + gt[0, 1] ** 2).numpy()
    return FLIP(crop(mag) * float(batch.scale))


def zero_filled_of(batch):
    x0 = batch.nw_input.float()
    mag = torch.sqrt(x0[0, 0] ** 2 + x0[0, 1] ** 2).numpy()
    return FLIP(crop(mag) * float(batch.scale))


def volume_name(batch):
    f = batch.fname
    return str(f[0] if isinstance(f, (list, tuple)) else f)


class Recons:

    def __init__(self, paths):
        self.files = {n: h5py.File(p, "r") for n, p in paths.items()}
        self.names = list(self.files)

    def get(self, name, vol, sl):
        f = self.files[name]
        if vol not in f or sl >= len(f[vol]):
            return None
        return FLIP(crop(np.asarray(f[vol])[sl]).astype(np.float64))

    def close(self):
        for f in self.files.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
