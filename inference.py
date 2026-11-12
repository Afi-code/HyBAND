import argparse
import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Config, build_loader
from hyband.neighbours import add_neighbours
from hyband.trainer import build_model
from hyband.infer import score_crop, mag
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-path", type=Path, default=None,
                    help="default: paths.DATA (env HYBAND_DATA)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--vols", type=int, default=0,
                    help="reconstruct only the FIRST N volumes (0 = all)")
    ap.add_argument("--volume", action="append", default=None, metavar="NAME",
                    help="reconstruct only this volume, e.g. --volume file1000031 "
                         "(repeatable). Overrides --vols.")
    ap.add_argument("--slices", default=None, metavar="SPEC",
                    help="only these slice indices within each volume: "
                         "'7', '0,7,12' or a range '5-9'. Default: all slices.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--radial", action="store_true")
    ap.add_argument("--radial-acs", type=int, default=24)
    a = ap.parse_args()
    data_path = a.data_path or paths.DATA
    want_vol = {v if v.endswith(".h5") else v + ".h5" for v in (a.volume or [])} or None
    want_sl = None
    if a.slices:
        want_sl = set()
        for part in a.slices.split(","):
            if "-" in part:
                lo, hi = part.split("-"); want_sl |= set(range(int(lo), int(hi) + 1))
            else:
                want_sl.add(int(part))

    if a.radial:
        spec = importlib.util.spec_from_file_location("_radial", HERE / "radial.py")
        rad = importlib.util.module_from_spec(spec); spec.loader.exec_module(rad)
        import data as D
        _orig = D.build_mask_func

        def _patched(cfg):
            if getattr(cfg, "mask_type", "") == "radial":
                print(f"RADIAL MASK: benchmark RadialMaskFunc, "
                      f"accel={list(cfg.accelerations)}, acs={a.radial_acs}", flush=True)
                return rad.RadialMaskFunc(list(cfg.accelerations), a.radial_acs)
            return _orig(cfg)
        D.build_mask_func = _patched

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = Namespace(**ck["args"])
    m = build_model(ta, dev)
    m.load_state_dict(ck["model"]); m.eval()
    cfg = Config(data_path=data_path, dataset=ta.dataset, challenge=ta.challenge,
                 accelerations=[ta.accel], center_fractions=[ta.center_fraction],
                 mask_type=ta.mask_type, batch_size=1, num_workers=a.num_workers,
                 smaps_cache=None, use_dataset_cache=False, ssdu_split=False,
                 rho=ta.rho, ssdu_mask_type=ta.ssdu_mask_type,
                 return_trn_kspace=bool(m.hard_lf_dc))
    ds = add_neighbours(build_loader(cfg, a.split, shuffle=False, device=dev).dataset,
                        cfg, m.nbr_slices)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.num_workers)

    vols, seen, n = {}, [], 0
    with torch.no_grad():
        for b in loader:
            fn = Path(str(b.fname[0])).stem + ".h5"
            sl_i = int(b.slice_num[0])
            if want_vol is not None and fn not in want_vol:
                continue
            if want_sl is not None and sl_i not in want_sl:
                continue
            if fn not in seen:
                if a.vols and want_vol is None and len(seen) >= a.vols:
                    break
                seen.append(fn)
            img, _ = m(b.nw_input.to(dev).float(), b.smaps.to(dev),
                       b.omega_mask.to(dev), b.omega_mask.to(dev),
                       b.trn_kspace.to(dev) if m.hard_lf_dc else None,
                       b.nbr_inputs.to(dev).float() if m.nbr_slices else None)
            sc = b.scale.view(-1, 1, 1)
            rec = score_crop(mag(img.cpu()) * sc)[0].numpy().astype(np.float32)
            vols.setdefault(fn, []).append((sl_i, rec))
            n += 1
            if n % 200 == 0:
                print(f"  {n} slices, {len(vols)} volumes", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        for fn, sl in sorted(vols.items()):
            sl.sort(key=lambda t: t[0])
            f.create_dataset(fn, data=np.stack([x[1] for x in sl]))
    if not vols:
        raise SystemExit(
            f"nothing matched: --volume {sorted(want_vol) if want_vol else 'any'} "
            f"--slices {a.slices or 'all'}. Volume names look like 'file1000031'.")
    print(f"wrote {a.out}: {len(vols)} volumes, {n} slices "
          f"({', '.join(f'{k} s{[i for i,_ in v]}' for k, v in sorted(vols.items()))[:160]})",
          flush=True)


if __name__ == "__main__":
    main()
