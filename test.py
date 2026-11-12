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
        print("$ " + " ".join(cmd), flush=True)
        if subprocess.call(cmd, cwd=HERE) != 0:
            sys.exit("reconstruction failed")

    cmd = [sys.executable, str(HERE / "eval_dump.py"),
           "--recon", str(a.recon), "--tag", a.tag, "--split", a.split,
           "--num-workers", str(a.num_workers), "--out-dir", str(a.out_dir)]
    if a.vols and not a.volume:
        cmd += ["--limit", str(a.vols)]
    print("\n$ " + " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=HERE))


if __name__ == "__main__":
    main()
