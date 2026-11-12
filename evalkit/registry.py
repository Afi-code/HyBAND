from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
PROJ = PKG_ROOT.parent
import sys as _sys
if str(PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PKG_ROOT))
import paths as _paths
HYB = PROJ / "ssdu-mamba_hybrid_hf"
R = "dataandoutputview/recons"

DATA = _paths.DATA
ROOT = _paths.TEST_ROOT
ANNOTATIONS = _paths.ANNOTATIONS


ACCEL = 8
ACS_LINES = 24
DATASET = "pdc30"
CROP = 320
EDGE = 16
