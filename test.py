import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--recon", type=Path, default=None,
                    help="score this existing .h5 instead of reconstructing")
    ap.add_argument("--tag", default="HyBAND")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--vols", type=int, default=0, help="0 = all; --vols 1 is a quick check")
    ap.add_argument("--volume", action="append", default=None,
                    help="only this volume, e.g. --volume file1000031 (repeatable)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pad-width", type=int, default=None)
    ap.add_argument("--out-cut", type=float, default=None)
    ap.add_argument("--keep-measured", type=int, default=None)
    ap.add_argument("--radial", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print both commands and exit WITHOUT running them")
    a = ap.parse_args()

    if a.recon is None:
        if a.ckpt is None:
            sys.exit("need --ckpt (to reconstruct) or --recon (to score an existing dump)")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        a.recon = a.out_dir / f"{a.tag}_{a.split}.h5"
        cmd = [sys.executable, str(HERE / "inference.py"),
               "--ckpt", str(a.ckpt), "--out", str(a.recon),
               "--split", a.split, "--num-workers", str(a.num_workers)]
        if a.vols:                   cmd += ["--vols", str(a.vols)]
        for v in (a.volume or []):   cmd += ["--volume", v]
        if a.pad_width is not None:  cmd += ["--pad-width", str(a.pad_width)]
        if a.out_cut is not None:    cmd += ["--out-cut", str(a.out_cut)]
        if a.keep_measured is not None: cmd += ["--keep-measured", str(a.keep_measured)]
        if a.radial:                 cmd += ["--radial"]
        print("$ " + " ".join(cmd), flush=True)
        if not a.dry_run and subprocess.call(cmd, cwd=HERE) != 0:
            sys.exit("reconstruction failed")

    cmd = [sys.executable, str(HERE / "eval_dump.py"),
           "--recon", str(a.recon), "--tag", a.tag, "--split", a.split,
           "--num-workers", str(a.num_workers), "--out-dir", str(a.out_dir)]
    if a.vols and not a.volume:
        cmd += ["--limit", str(a.vols)]
    print("\n$ " + " ".join(cmd), flush=True)
    if a.dry_run:
        print("(--dry-run: not executed)")
        sys.exit(0)
    sys.exit(subprocess.call(cmd, cwd=HERE))


if __name__ == "__main__":
    main()
