import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--accel", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", default="5e-4")
    ap.add_argument("--batch-size", type=int, default=2)
    a = ap.parse_args()
    data = a.data_path or str(paths.DATA)

    env = dict(os.environ)
    cmd = [sys.executable, "-u", "-m", "hyband.trainer", "--arch", "hyband",
           "--data-path", data, "--dataset", "pdc30", "--challenge", "multicoil",
           "--accel", str(a.accel),
           "--center-fraction", "0.0645",
           "--mask-type", "equispaced",
           "--rho", "0.4", "--ssdu-mask-type", "Gaussian",
           "--band-c1", "0.06", "--band-c2", "0.20", "--band-axis", "radial",
           "--m-features", "80", "--d-state", "32",
           "--h-ch", "32", "--h-depth", "2",
           "--th-pad", "16", "--th-supp-thresh", "1e-4",
           "--th-mu", "0.100605", "--th-anchor", "confined", "--th-hf-source", "x",
           "--nb-unroll-blocks", "10", "--cg-iter", "10",
           "--checkpoint-unrolls", "--checkpoint-stride", "1",
           "--lr", str(a.lr), "--batch-size", str(a.batch_size),
           "--grad-clip", "1.0", "--max-epochs", str(a.epochs),
           "--select", "psnr", "--freeze-routes", "",
           "--null-target", "0", "--out-dir", a.out_dir,
           "--m-pattern", "MCMCMC", "--loss", "band", "--band-weights", "1,1,1",
           "--nbr-slices", "1",
           "--th-out-cut", "1", "--th-keep-measured-corners",
           "--null-weight", "1"]

    print("$ " + " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=HERE, env=env))


if __name__ == "__main__":
    main()
