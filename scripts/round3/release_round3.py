"""Refresh ../anonymous_release with round-3 code and results, write onset_events.csv, and re-scan for identity clues.

    python scripts/round3/release_round3.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from anon_clues import load_clues  # noqa: E402

REL = ROOT.parent / "anonymous_release"
CLUES = load_clues(ROOT)  # generic patterns + local .anon_clues (git-ignored)


def onset_events():
    tr = pd.read_csv("results/round3/onset_trials.csv.gz")
    tr = tr[tr.detector.isin(["lda", "riemann"]) & (tr.threshold == tr.operating_threshold)]
    ev = pd.DataFrame({"subject": tr.subject, "split": tr.scheme, "fold": tr.fold, "trial_id": tr.trial_id,
                       "LED_on": tr.led_on, "hand_start": tr.hand_start, "detector": tr.detector,
                       "threshold": tr.threshold, "first_detection_time": tr.first_detection_time,
                       "strict_pre": tr.strict_pre, "loose_hit": tr.loose_hit, "rest_seconds": tr.rest_seconds,
                       "n_false_alarms": tr.n_false_alarms})
    cue = pd.read_csv("results/round3/cue_timer_trials.csv")
    cev = pd.DataFrame({"subject": cue.subject, "split": cue.scheme, "fold": cue.fold, "trial_id": cue.trial_id,
                        "LED_on": cue.led_on, "hand_start": cue.hand_start, "detector": "cue_timer", "threshold": np.nan,
                        "first_detection_time": cue.led_on + 0.1, "strict_pre": cue.strict_pre, "loose_hit": cue.loose_hit,
                        "rest_seconds": np.nan, "n_false_alarms": 0})
    out = pd.concat([ev, cev], ignore_index=True)
    out.to_csv(REL / "results" / "onset_events.csv", index=False)
    return len(out)


def main():
    for d in ("src", "scripts", "configs", "tests"):
        shutil.rmtree(REL / d, ignore_errors=True)
        shutil.copytree(ROOT / d, REL / d, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc"))
    for f in ("make_anonymous_release.py",):
        (REL / "scripts" / f).unlink(missing_ok=True)
    (REL / "scripts" / "round3" / "release_round3.py").unlink(missing_ok=True)
    for f in ("README.md", "pyproject.toml", "environment.yml", "Makefile"):
        shutil.copy(ROOT / f, REL / f)
    for f in ("DATA_CARD.md", "DECISIONS.md", "CHECKLIST.md"):
        shutil.copy(ROOT / "docs" / f, REL / "docs" / f)
    dst = REL / "results" / "round3"
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path("results/round3").glob("*"):
        if f.suffix in (".csv", ".gz", ".md"):
            shutil.copy(f, dst / f.name)
    n = onset_events()
    hits = []
    for p in REL.rglob("*"):
        if p.is_file() and p.name != "anon_clues.py":  # the scanner's own pattern list
            txt = p.read_bytes()
            try:
                import gzip

                txt = gzip.decompress(txt) if p.suffix == ".gz" else txt
            except OSError:
                pass
            low = txt.decode("utf-8", errors="ignore").lower()
            hits += [(str(p.relative_to(REL)), c) for c in CLUES if c in low]
    print(f"onset_events rows: {n}; release files: {sum(1 for p in REL.rglob('*') if p.is_file())}")
    print("identity-clue hits:", hits if hits else "none")


if __name__ == "__main__":
    main()
