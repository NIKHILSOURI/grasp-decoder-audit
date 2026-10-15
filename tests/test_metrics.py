import numpy as np

from gaudit.eval.metrics import pearson, r2, rmse


def test_r2_perfect_and_mean():
    y = np.random.default_rng(0).normal(size=1000)
    assert r2(y, y) == 1.0
    assert abs(r2(y, np.full_like(y, y.mean()))) < 1e-12
    assert rmse(y, y) == 0.0


def test_r_can_be_high_while_r2_is_negative():
    y = np.sin(np.linspace(0, 6, 500))
    p = 3 * y + 5  # perfectly correlated but badly scaled and offset
    assert pearson(y, p) > 0.999 and r2(y, p) < 0
