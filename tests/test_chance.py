import numpy as np

from gaudit.experiments import _null_targets


def _toy(n_trials=12, seed=0):
    g = np.random.default_rng(seed)
    tid, k, y = [], [], []
    for t in range(n_trials):
        n = int(g.integers(20, 40))
        tid += [1000 + t] * n
        k += list(range(n))
        y += [float(t)] * n  # the target value encodes its trial
    return np.array(tid), np.array(k), np.array(y)


def test_mismatch_never_uses_own_trial():
    tid, k, y = _toy()
    for seed in range(50):
        ym = _null_targets(tid, k, y, np.random.default_rng(seed), "mismatch")
        assert not np.any(ym == tid - 1000), "a trial was paired with its own target"


def test_reverse_is_time_reversal_within_trial():
    tid, k, _ = _toy()
    y = k.astype(float)
    yr = _null_targets(tid, k, y, None, "reverse")
    for t in np.unique(tid):
        m = tid == t
        assert np.array_equal(yr[m], y[m][::-1])
