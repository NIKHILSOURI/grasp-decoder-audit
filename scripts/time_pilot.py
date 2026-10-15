"""Timing pilot: time one fold per (model, scheme) on a few subjects and project full E1 cost.

    python scripts/time_pilot.py [--subjects 1 2 3 4]
Writes results/pilot/timing.json and prints the projection used for budget decisions.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

from gaudit.data.splits import make_splits
from gaudit.pipeline import Data, Pool, model_spec, run_fold

warnings.filterwarnings("ignore")
N_FOLDS = {"shuffled": 5, "trialwise": 5, "serieswise": 3}  # per subject, with default repeats
N_FOLDS_EEGNET = {"shuffled": 3, "trialwise": 3, "serieswise": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=["template", "ridge", "mlp", "riemann", "eegnet", "forcemap"])
    a = ap.parse_args()
    data = Data("data/cache", a.subjects)
    res, proj = {}, {}
    for ds, tg in (("force", ["force"]), ("kin", ["pos_x", "pos_y", "pos_z"])):
        for m in a.models:
            if m == "forcemap" and ds != "force":
                continue
            spec = model_spec(m)
            pool = Pool(data, a.subjects, ds, spec.win, spec.stride)
            t0 = time.perf_counter()
            pool.feature(spec.feature)
            feat_s = time.perf_counter() - t0
            total = feat_s * 12 / len(a.subjects)
            for scheme in ("shuffled", "trialwise", "serieswise", "subjectwise"):
                sp = [s for s in make_splits(pool.w, scheme, a.subjects, 0, n_repeats=1)
                      if s.test_subjects == (a.subjects[0],)][0]
                t0 = time.perf_counter()
                run_fold(pool, sp, m, tg, 0, model_kw={"eegnet": {"max_epochs": 40}}.get(m))
                dt = time.perf_counter() - t0
                res[f"{ds}/{m}/{scheme}"] = dt
                if scheme == "subjectwise":  # training set grows ~ 9/(len-3) times with 12 subjects
                    total += dt * 12 * 9 / max(1, len(a.subjects) - 3)
                else:
                    total += dt * 12 * (N_FOLDS_EEGNET if m == "eegnet" else N_FOLDS)[scheme]
                print(f"{ds:5s} {m:8s} {scheme:11s} {dt:6.1f}s")
            proj[f"{ds}/{m}"] = total / 60
    print("\nprojected full E1 minutes:", {k: round(v, 1) for k, v in proj.items()},
          "total", round(sum(proj.values()), 1))
    Path("results/pilot").mkdir(parents=True, exist_ok=True)
    Path("results/pilot/timing.json").write_text(json.dumps({"fold_s": res, "projected_min": proj}, indent=2))


if __name__ == "__main__":
    main()
