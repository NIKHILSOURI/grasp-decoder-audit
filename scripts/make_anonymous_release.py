"""Build an anonymized release folder (code + split manifests + per-participant results) for upload.

    python scripts/make_anonymous_release.py  ->  ../anonymous_release/
The folder contains no data files, no git history, no logs with local paths, and is scanned for identity
clues. Shuffled and blocked splits are window-level and regenerated deterministically by the code (seed 0).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from gaudit.data.splits import make_splits, split_masks
from gaudit.experiments import latest_run, load_e1
from gaudit.pipeline import Data, Pool

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from anon_clues import load_clues  # noqa: E402

OUT = ROOT.parent / "anonymous_release"
S = list(range(1, 13))
CLUES = load_clues(ROOT)  # generic patterns + local .anon_clues (git-ignored)


def manifests(dst: Path):
    data = Data("data/cache", S)
    rows = []
    for ds in ("force", "kin"):
        for stride, reps, tag in ((5, 5, "default"), (10, 3, "eegnet")):
            pool = Pool(data, S, ds, 50, stride)
            for scheme in ("trialwise", "serieswise", "subjectwise"):
                if tag == "eegnet" and scheme != "trialwise":
                    continue  # identical trial assignment; only the number of repeats differs
                for sp in make_splits(pool.w, scheme, S, 0, n_repeats=reps):
                    tr, va, te = split_masks(pool.w, sp)
                    for part, m in (("train", tr), ("validation", va), ("test", te)):
                        for tid in sorted(set(pool.w.trial_id[m].tolist())):
                            rows.append({"dataset": ds, "scheme": scheme, "fold": sp.fold, "repeats_set": tag,
                                         "partition": part, "trial_id": tid})
    df = pd.DataFrame(rows).drop_duplicates()
    df["participant"] = df.trial_id // 10000
    df["series"] = (df.trial_id // 100) % 100
    df["lift"] = df.trial_id % 100
    df.to_csv(dst / "split_manifest_trials.csv", index=False)
    return len(df)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for d in ("src", "scripts", "configs", "tests"):
        shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc"))
    (OUT / "docs").mkdir()
    for f in ("DATA_CARD.md", "DECISIONS.md", "CHECKLIST.md"):
        shutil.copy(ROOT / "docs" / f, OUT / "docs" / f)
    for f in ("README.md", "pyproject.toml", "environment.yml", "Makefile"):
        shutil.copy(ROOT / f, OUT / f)
    (OUT / "scripts" / "make_anonymous_release.py").unlink()
    # per-participant results of the final runs (CSV only; no logs or configs with local paths)
    res = OUT / "results"
    res.mkdir()
    sub, _ = load_e1()
    sub.drop(columns=["run"]).to_csv(res / "e1_per_subject.csv", index=False)
    for name in ("e2_selection", "e3_chance", "e4_onset", "e5_artifacts", "e6_blocks", "e7_residual"):
        run = latest_run(name)
        for f in run.glob("*.csv"):
            shutil.copy(f, res / f"{name.split('_')[0]}_{f.name}")
    for f in Path("results/tables").glob("*"):
        shutil.copy(f, res / f"table_{f.name}")
    (OUT / "splits").mkdir()
    n = manifests(OUT / "splits")
    (OUT / "splits" / "README.md").write_text(
        "split_manifest_trials.csv lists, for every honest split (trial-wise, series-wise, subject-wise), the trial IDs\n"
        "in train / validation / test per fold. Trial ID = participant*10000 + series*100 + lift.\n"
        "Shuffled-window and blocked splits are window-level; `src/gaudit/data/splits.py` regenerates them exactly\n"
        "(seed 0). EEGNet uses repeats r0-r2 of the trial-wise split.\n", encoding="utf-8")
    # identity scan
    hits = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.name != "anon_clues.py":  # the scanner's own pattern list
            txt = p.read_text(encoding="utf-8", errors="ignore").lower()
            for c in CLUES:
                if c in txt:
                    hits.append((str(p.relative_to(OUT)), c))
    print(f"release: {sum(1 for p in OUT.rglob('*') if p.is_file())} files, manifest rows {n}")
    print("identity-clue hits:", hits if hits else "none")


if __name__ == "__main__":
    main()
