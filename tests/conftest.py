import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
needs_cache = pytest.mark.skipif(not (CACHE / "P5.npz").exists(), reason="run `make preprocess` first")


@pytest.fixture(scope="session")
def data():
    from gaudit.pipeline import Data

    return Data(str(CACHE), [1, 2, 3, 4, 5])


@pytest.fixture(scope="session")
def pool(data):
    from gaudit.pipeline import Pool

    return Pool(data, [1, 2, 3, 4, 5], "force", 50, 5)
