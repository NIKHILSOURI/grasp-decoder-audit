"""Run-folder outputs: config, git hash, seed, library versions, GPU name."""

import json
import platform
import subprocess
from pathlib import Path


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    import numpy
    import pyriemann
    import scipy
    import sklearn
    import torch

    info = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "pyriemann": pyriemann.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform()}
    try:
        import braindecode

        info["braindecode"] = braindecode.__version__
    except Exception:
        pass
    return info


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
