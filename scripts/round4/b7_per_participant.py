"""Per-participant values for the round-3 B7 result (MLP, force, 1-s windows, trial-wise and shuffled), needed for
the traceability table (A2). Same code path, splits and seeds as scripts/round3/b2_b7_filtering_mlp.py.

    python scripts/round4/b7_per_participant.py -> results/round4/b7_per_participant.csv
"""

from __future__ import annotations

import warnings

import pandas as pd

from gaudit.data.splits import make_splits
from gaudit.experiments import fold_seed
from gaudit.pipeline import Data, Pool, run_fold
from gaudit.utils.seed import seed_everything

warnings.filterwarnings("ignore")
S = list(range(1, 13))


def main():
    data = Data("data/cache", S)
    pool = Pool(data, S, "force", 100, 5)
    rows = []
    for scheme in ("trialwise", "shuffled"):
        for sp in make_splits(pool.w, scheme, S, 0, n_repeats=5):
            seed_everything(fold_seed(0, "force", "mlp", scheme, sp.fold))
            r, _ = run_fold(pool, sp, "mlp", ["force"], fold_seed(0, "mlp", sp.fold),
                            model_kw={"max_epochs": 200, "patience": 10, "batch": 512})
            for x in r:
                x["window_s"] = 1.0
            rows += r
        print(scheme, "done", flush=True)
    df = pd.DataFrame(rows).drop(columns=["info"])
    df.to_csv("results/round4/b7_folds.csv", index=False)
    df.groupby(["scheme", "subject"], as_index=False).r2.mean().to_csv("results/round4/b7_per_participant.csv", index=False)


if __name__ == "__main__":
    main()
