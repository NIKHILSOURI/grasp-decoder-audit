"""Print the structure of the raw WAY-EEG-GAL MATLAB files (used to write docs/DATA_CARD.md)."""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


def show(obj, name="", depth=0, maxdepth=3):
    pad = "  " * depth
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        print(f"{pad}{name}: struct array shape={obj.shape} fields={obj.dtype.names}")
        if depth < maxdepth and obj.size:
            el = obj.flat[0]
            for f in obj.dtype.names:
                show(el[f], f, depth + 1, maxdepth)
    elif isinstance(obj, np.ndarray):
        s = ""
        if obj.dtype.kind in "fiu" and obj.size:
            s = f" min={np.nanmin(obj):.4g} max={np.nanmax(obj):.4g}"
        elif obj.dtype == object and obj.size:
            s = f" first={obj.flat[0]!r}"[:200]
        elif obj.dtype.kind == "U":
            s = f" value={obj!r}"[:200]
        print(f"{pad}{name}: {obj.dtype} shape={obj.shape}{s}")
    else:
        print(f"{pad}{name}: {type(obj).__name__} {obj!r}"[:200])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("=" * 80, "\n", p, f"{Path(p).stat().st_size / 1e6:.1f} MB")
        d = sio.loadmat(p)
        for k, v in d.items():
            if not k.startswith("__"):
                show(v, k)
