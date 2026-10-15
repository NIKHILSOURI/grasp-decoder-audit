import numpy as np

from gaudit.data.preprocess import online_filters
from gaudit.onset import detector as od


def test_causal_filters_ignore_the_future():
    x = np.random.default_rng(0).normal(size=(5000, 32)).astype(np.float32)
    y = x.copy()
    y[3000:] += 500.0  # future-only spike
    a_bb, a_lf = online_filters(x)
    b_bb, b_lf = online_filters(y)
    assert np.allclose(a_bb[:3000], b_bb[:3000]) and np.allclose(a_lf[:3000], b_lf[:3000])


def _stream(n=3000, spike_at=None):
    g = np.random.default_rng(1)
    st = {"bb_S1": g.normal(size=(n, 32)).astype(np.float32), "lf_S1": g.normal(size=(n, 32)).astype(np.float32)}
    if spike_at is not None:
        st["bb_S1"][spike_at:] += 100
        st["lf_S1"][spike_at:] += 100
    return st


T = np.round(np.arange(1.0, 20.0, 0.02), 2)
SPIKE_T = 12.0


def test_features_use_only_past_samples():
    S = np.ones(len(T), int)
    a, b = _stream(), _stream(spike_at=int(SPIKE_T * 100) + 1)
    past = T <= SPIKE_T
    for kind in ("lda", "riemann"):
        fa, fb = od.features(a, S, T, kind), od.features(b, S, T, kind)
        assert np.allclose(fa[past], fb[past]), kind
        assert not np.allclose(fa[~past], fb[~past]), kind


def test_no_detection_before_a_future_only_spike():
    S = np.ones(len(T), int)
    X = od.features(_stream(spike_at=int(SPIKE_T * 100) + 1), S, T, "lda")
    p = 1 / (1 + np.exp(-(X[:, :32].mean(1) - 50)))  # toy classifier: fires on large amplitude
    ev = T[od.detection_events(p, 0.5)]
    assert ev.size and ev.min() > SPIKE_T


def test_detection_rule_needs_three_consecutive_steps():
    p = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    assert od.detection_events(p, 0.5).tolist() == [5, 10]
