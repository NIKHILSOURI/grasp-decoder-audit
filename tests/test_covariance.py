import numpy as np
from pyriemann.geometry.mean import mean_riemann as pr_mean
from scipy.linalg import logm

from gaudit.features.covariance import TangentSpace, mean_riemann, oas_cov


def _covs(n=300):
    return oas_cov(np.random.default_rng(0).normal(size=(n, 8, 50)).astype(np.float32))


def test_oas_matches_sklearn():
    from sklearn.covariance import oas

    x = np.random.default_rng(1).normal(size=(3, 8, 50))
    ours = oas_cov(x)
    for i in range(3):
        ref, _ = oas(x[i].T, assume_centered=False)
        assert np.allclose(ours[i], ref, rtol=1e-5, atol=1e-8)


def test_karcher_mean_matches_pyriemann():
    C = _covs()
    A, B = pr_mean(C), mean_riemann(C)
    assert np.abs(A - B).max() / np.abs(A).max() < 1e-6


def test_tangent_space_matches_scipy_logm():
    C = _covs()
    ts = TangentSpace().fit(C)
    T = ts.transform(C)
    L = logm(ts.isqrt_ @ C[7] @ ts.isqrt_).real
    assert np.abs(L[ts.iu_] * ts.coef_ - T[7]).max() < 1e-3
