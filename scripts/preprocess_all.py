"""Preprocess every participant once and cache float32 arrays to data/cache/P{n}.npz."""

import argparse
import json
import time
from pathlib import Path

from joblib import Parallel, delayed

from gaudit.data.cache import SUBJECTS, build_subject, load_cached, save_subject


def one(raw, cache, s, force):
    p = Path(cache) / f"P{s}.npz"
    t0 = time.time()
    if force or not p.exists():
        save_subject(cache, s, *build_subject(raw, s))
    d = load_cached(cache, s)  # re-load and run all shape/ID assertions
    return s, len(d.trial_id), int(d.bb.shape[0]), round(time.time() - t0, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--subjects", type=int, nargs="*", default=SUBJECTS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs)(delayed(one)(a.raw, a.cache, s, a.force) for s in a.subjects)
    for s, nt, ns, dt in res:
        print(f"P{s}: {nt} trials, {ns} samples @100 Hz, {dt}s")
    total = time.time() - t0
    print(f"total {total:.0f}s")
    Path("results").mkdir(exist_ok=True)
    Path("results/preprocess_timing.json").write_text(json.dumps({"seconds": total, "subjects": res}))
