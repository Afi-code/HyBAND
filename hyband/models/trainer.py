import argparse
from pathlib import Path

from hyband.losses import BandNormalizedKLoss, MixedL1L2Loss

MASK_TYPES = ["equispaced", "radial"]

ARCH_ALIASES = {"hyband": "hyband"}


def arch_of(args):
    return ARCH_ALIASES.get(getattr(args, "arch", "hyband"), "hyband")


def get_args():
    a = argparse.ArgumentParser(description="HyBAND training")
    a.add_argument("--arch", default="hyband", choices=list(ARCH_ALIASES))
    a.add_argument("--data-path", type=Path, default=Path("hyband/data_root"))
    a.add_argument("--dataset", default="pdc30")
    a.add_argument("--challenge", default="multicoil")
    a.add_argument("--out-dir", type=Path, required=True)

    a.add_argument("--accel", type=int, default=8)
    a.add_argument("--center-fraction", type=float, default=0.0645,
                   help="ACS width as a fraction of the phase-encode columns; "
                        "0 for the radial masks")
    a.add_argument("--mask-type", default="equispaced", choices=MASK_TYPES)
    a.add_argument("--rho", type=float, default=0.4,
                   help="SSDU: fraction of Omega held out as Lambda")
    a.add_argument("--ssdu-mask-type", default="Gaussian")

    a.add_argument("--band-c1", type=float, default=0.06)
    a.add_argument("--band-c2", type=float, default=0.20)
    a.add_argument("--band-axis", default="radial")
    a.add_argument("--band-transition", type=float, default=0.25)

    a.add_argument("--m-features", type=int, default=80)
    a.add_argument("--m-pattern", default="MCMCMC",
                   help="prior blocks: M = Mamba (VSS), C = ConvNeXt")
    a.add_argument("--d-state", type=int, default=32)
    a.add_argument("--h-ch", type=int, default=32)
    a.add_argument("--h-depth", type=int, default=2)
    a.add_argument("--nbr-slices", type=int, default=1)

    a.add_argument("--th-pad", type=int, default=16)
    a.add_argument("--th-supp-thresh", type=float, default=1e-4)
    a.add_argument("--th-mu", type=float, default=0.100605)
    a.add_argument("--th-anchor", default="confined")
    a.add_argument("--th-hf-source", default="x")
    a.add_argument("--th-out-cut", type=float, default=1.0)
    a.add_argument("--th-keep-measured-corners", action="store_true")
    a.add_argument("--nb-unroll-blocks", type=int, default=10)
    a.add_argument("--cg-iter", type=int, default=10)

    a.add_argument("--loss", default="band", choices=["band", "global"])
    a.add_argument("--band-weights", default="1,1,1")
    a.add_argument("--null-weight", type=float, default=1.0,
                   help="weight on the null-space growth penalty")

    a.add_argument("--lr", type=float, default=5e-4)
    a.add_argument("--batch-size", type=int, default=2)
    a.add_argument("--max-epochs", type=int, default=40)
    a.add_argument("--grad-clip", type=float, default=1.0)
    a.add_argument("--num-workers", type=int, default=8)
    a.add_argument("--checkpoint-unrolls", action="store_true")
    a.add_argument("--checkpoint-stride", type=int, default=1)
    a.add_argument("--select", default="psnr")
    a.add_argument("--resume", type=Path, default=None)
    a.add_argument("--seed", type=int, default=0)

    args = a.parse_args()
    args.arch = ARCH_ALIASES[args.arch]
    return args


def make_loss(args):
    bw = tuple(float(v) for v in args.band_weights.split(","))
    return (BandNormalizedKLoss(c1=args.band_c1, c2=args.band_c2,
                                transition=args.band_transition,
                                axis=args.band_axis, weights=bw)
            if args.loss == "band" else MixedL1L2Loss()), bw
