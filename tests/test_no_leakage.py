"""Spy on every fitted step: in honest schemes nothing may be fit on test windows.

A sentinel value is planted in the features and targets of every test row; spies on the
target scaler, feature scalers, top-k selection and the Riemannian reference fail the test
if they ever receive a sentinel.
"""

import numpy as np
import pytest
from conftest import needs_cache

import gaudit.features.covariance as cov_mod
import gaudit.models.base as base_mod
import gaudit.models.mlp as mlp_mod
import gaudit.models.ridge as ridge_mod
import gaudit.models.riemann as rie_mod
import gaudit.pipeline as pl
from gaudit.data.splits import make_splits, split_masks
from gaudit.data.targets import TargetScaler

pytestmark = needs_cache
SENT = 7.77e7
S = [1, 2, 3, 4, 5]


class Spy:
    def __init__(self):
        self.calls = []

    def check(self, name, *arrays):
        hit = any(np.isclose(np.asarray(a, float), SENT, rtol=0, atol=1).any() for a in arrays)
        self.calls.append((name, bool(hit)))


@pytest.fixture
def spied(monkeypatch):
    spy = Spy()
    real_scaler = base_mod.make_scaler

    class SpyScaler:
        def __init__(self):
            self.s = real_scaler()

        def fit(self, X, y=None):
            spy.check("feature_scaler", X)
            self.s.fit(X)
            return self

        def transform(self, X):
            return self.s.transform(X)

    real_ts_fit, real_rank, real_tfit = cov_mod.TangentSpace.fit, base_mod.corr_rank, TargetScaler.fit

    def ts_fit(self, C):
        spy.check("riemann_reference", C)
        return real_ts_fit(self, C)

    def rank(X, y, k):
        spy.check("selection", X, y)
        return real_rank(X, y, k)

    def tfit(self, y):
        spy.check("target_scaler", y)
        return real_tfit(self, y)

    for m in (base_mod, mlp_mod, ridge_mod, rie_mod):
        monkeypatch.setattr(m, "make_scaler", SpyScaler)
    monkeypatch.setattr(cov_mod.TangentSpace, "fit", ts_fit)
    monkeypatch.setattr(base_mod, "corr_rank", rank)
    monkeypatch.setattr(pl, "corr_rank", rank)
    monkeypatch.setattr(TargetScaler, "fit", tfit)
    return spy


def _plant(pool, sp, kind, monkeypatch):
    """Register a feature copy whose test rows carry the sentinel; same for the targets."""
    _, _, te = split_masks(pool.w, sp)
    X = pool.feature(kind).copy()
    X[te] = np.eye(X.shape[1], dtype=X.dtype) * SENT if kind == "cov" else SENT
    pool._feat[(kind + "_planted", ())] = X
    y_real = pool.y.__func__

    def y(names):
        out = y_real(pool, names).copy()
        out[te] = SENT
        return out

    monkeypatch.setattr(pool, "y", y)


@pytest.mark.parametrize("scheme", ["trialwise", "serieswise", "subjectwise"])
@pytest.mark.parametrize("model,kind,select", [("ridge", "lagged", "train"), ("riemann", "cov", None),
                                               ("mlp", "bandpower", "train"), ("template", "time", None)])
def test_honest_fits_never_see_test_rows(spied, pool, monkeypatch, scheme, model, kind, select):
    sp = make_splits(pool.w, scheme, S, seed=0, n_repeats=1)[0]
    _plant(pool, sp, kind, monkeypatch)
    kw = {"max_epochs": 2, "patience": 1} if model == "mlp" else None
    with np.errstate(all="ignore"):
        pl.run_fold(pool, sp, model, ["force"], 0, model_kw=kw, select_on=select,
                    select_k=16 if select else None, feature_kind=kind + "_planted")
    assert spied.calls, "spies were not called"
    leaks = [name for name, hit in spied.calls if hit]
    assert not leaks, f"fitted on test rows: {leaks}"


def test_leaky_selection_does_see_test_rows(spied, pool, monkeypatch):
    sp = make_splits(pool.w, "trialwise", S, seed=0, n_repeats=1)[0]
    _plant(pool, sp, "lagged", monkeypatch)
    with np.errstate(all="ignore"):
        pl.run_fold(pool, sp, "ridge", ["force"], 0, select_on="all", select_k=16, feature_kind="lagged_planted")
    assert any(hit for name, hit in spied.calls if name == "selection"), "select_on=all must include test rows"


def test_shuffled_scheme_shares_trials(pool):
    sp = make_splits(pool.w, "shuffled", S, seed=0, n_repeats=1)[0]
    tr, _, te = split_masks(pool.w, sp)
    assert np.intersect1d(pool.w.trial_id[tr], pool.w.trial_id[te]).size > 0, "shuffled must be the leaky one"


@pytest.mark.parametrize("scheme", ["trialwise", "serieswise", "subjectwise"])
def test_honest_schemes_share_no_trials_or_subjects(pool, scheme):
    for sp in make_splits(pool.w, scheme, S, seed=0, n_repeats=1):
        tr, va, te = split_masks(pool.w, sp)
        assert not np.intersect1d(pool.w.trial_id[tr], pool.w.trial_id[te]).size
        if scheme == "subjectwise":
            assert not np.intersect1d(pool.w.subject[tr], pool.w.subject[te]).size
