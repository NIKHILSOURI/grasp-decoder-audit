"""Experiment engine: window pools with cached features, and a single-fold runner.

A *pool* is the set of windows of one dataset ('force' = constant-surface trials,
'kin' = all trials) over a list of subjects, at one window length and stride. Features are
stateless per window, so they are computed once per pool and sliced by fold. Everything
fitted (target scaler, feature scaler, selection, PCA, Riemannian reference, alpha, early
stopping) happens inside `run_fold` on training (and validation) rows only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .data.cache import SubjectData, load_cached, load_stream
from .data.splits import Split, split_masks
from .data.targets import TargetScaler, target_values
from .data.windows import Windows, make_windows, trial_mask
from .eval.metrics import all_metrics
from .features.bandpower import PREMOTOR_PARIETAL, bandpower
from .features.covariance import oas_cov
from .features.forcemap import FM_STRIDE, FM_WIN, forcemap_features
from .features.lagged import lagged, raw_windows
from .features.selection import corr_rank
from .models.eegnet import EEGNetDecoder
from .models.mlp import MLPDecoder
from .models.residual import TemplatePlus
from .models.ridge import RidgeDecoder
from .models.riemann import RiemannDecoder
from .models.template import TemplateDecoder

ARTIFACT_CHANNELS = ["Fp1", "Fp2", "O1", "Oz", "O2", "PO9", "PO10"]  # frontal-polar + occipital


@dataclass
class ModelSpec:
    name: str
    feature: str
    win: int
    stride: int


def model_spec(name: str, cfg_stride: dict | None = None) -> ModelSpec:
    st = cfg_stride or {}
    specs = {
        "template": ModelSpec("template", "time", 50, st.get("default", 5)),
        "ridge": ModelSpec("ridge", "lagged", 50, st.get("default", 5)),
        "mlp": ModelSpec("mlp", "bandpower", 50, st.get("default", 5)),
        "riemann": ModelSpec("riemann", "cov", 50, st.get("default", 5)),
        "eegnet": ModelSpec("eegnet", "raw", 50, st.get("eegnet", 10)),
        "forcemap": ModelSpec("forcemap", "forcemap", FM_WIN, FM_STRIDE),
    }
    if name.startswith("template+"):
        inner = specs[name.split("+", 1)[1]]
        return ModelSpec(name, "time+" + inner.feature, inner.win, inner.stride)
    return specs[name]


def build_model(name: str, seed: int, kw: dict | None = None, select_k=None, select_fixed=None,
                channels=None):
    kw = dict(kw or {})
    if name == "template":
        return TemplateDecoder()
    if name == "ridge":
        return RidgeDecoder(select_k=select_k, select_fixed=select_fixed)
    if name == "mlp":
        return MLPDecoder(select_k=select_k, select_fixed=select_fixed, seed=seed, **kw)
    if name == "riemann":
        return RiemannDecoder(seed=seed, channels=channels)
    if name == "eegnet":
        return EEGNetDecoder(seed=seed, **kw)
    if name == "forcemap":
        return MLPDecoder(feature="forcemap", pca=0.95, seed=seed, **kw)
    if name.startswith("template+"):  # template + EEG decoder on the residual (E7)
        return TemplatePlus(build_model(name.split("+", 1)[1], seed, kw))
    raise ValueError(name)


class Data:
    """All cached subjects, loaded once."""

    def __init__(self, cache_dir: str, subjects: list[int]):
        self.subjects = list(subjects)
        self.sd: dict[int, SubjectData] = {s: load_cached(cache_dir, s) for s in subjects}
        self.seg0 = {s: load_stream(cache_dir, s)["seg_start_abs"] for s in subjects}
        for s in subjects:
            assert len(self.seg0[s]) == len(self.sd[s].trial_id)


class Pool:
    def __init__(self, data: Data, subjects: list[int], dataset: str, win: int, stride: int):
        self.data, self.subjects, self.dataset, self.win, self.stride = data, list(subjects), dataset, win, stride
        parts = [make_windows(data.sd[s], trial_mask(data.sd[s], dataset), win, stride, data.seg0[s])
                 for s in subjects]
        self.w = Windows.concat(parts)
        self.w.check()
        bounds = np.cumsum([0] + [len(p) for p in parts])
        self.rows = {s: slice(bounds[i], bounds[i + 1]) for i, s in enumerate(subjects)}
        self._feat: dict = {}

    def _end(self, s):
        return self.w.end[self.rows[s]]

    def y(self, names: list[str]) -> np.ndarray:
        return np.concatenate([target_values(self.data.sd[s], self._end(s), names) for s in self.subjects])

    def feature(self, kind: str, drop: tuple = ()) -> np.ndarray:
        key = (kind, tuple(drop))
        if key in self._feat:
            return self._feat[key]
        if kind.startswith("time+"):  # [t_rel | EEG features] for TemplatePlus decoders
            X = np.hstack([self.feature("time"), self.feature(kind[5:], drop)])
            self._feat[key] = X
            return X
        out = []
        for s in self.subjects:
            sd, end = self.data.sd[s], self._end(s)
            ch = [i for i, c in enumerate(sd.ch_names) if c not in drop] if drop else None
            if kind == "lagged":
                f = lagged(sd.lf, end, channels=ch)
            elif kind == "bandpower":
                pp = [sd.ch_names.index(c) for c in PREMOTOR_PARIETAL if c not in drop]
                f = bandpower(raw_windows(sd.bb, end, self.win, channels=pp))
            elif kind == "cov":
                f = oas_cov(raw_windows(sd.bb, end, self.win, channels=ch)).astype(np.float32)
            elif kind == "raw":
                f = raw_windows(sd.bb, end, self.win, channels=ch)
            elif kind == "time":
                f = self.w.t_rel[self.rows[s]][:, None].astype(np.float32)
            elif kind == "forcemap":
                f = forcemap_features(sd, end)
            else:
                raise ValueError(kind)
            assert len(f) == len(end)
            out.append(f)
        X = np.concatenate(out)
        assert len(X) == len(self.w)
        self._feat[key] = X
        return X

    def drop_features(self, kind=None):
        for k in list(self._feat):
            if kind is None or k[0] == kind:
                del self._feat[k]


def run_fold(pool: Pool, split: Split, model: str, targets: list[str], seed: int, *, model_kw=None,
             select_on: str | None = None, select_k: int | None = None, drop: tuple = (),
             feature_kind: str | None = None):
    spec_feature = feature_kind or model_spec(model).feature
    tr, va, te = split_masks(pool.w, split)
    X = pool.feature(spec_feature, drop=drop)
    y = pool.y(targets)
    select_fixed = None
    if select_on == "all":  # LEAKY on purpose: selection sees test windows
        allm = tr | va | te
        select_fixed = corr_rank(X[allm], y[allm], select_k)
    elif select_on not in (None, "train"):
        raise ValueError(select_on)
    # dropped channels (E5) are removed when the features are built, so models need no channel list
    m = build_model(model, seed, model_kw, select_k=select_k if select_on == "train" else None,
                    select_fixed=select_fixed)
    ts = TargetScaler().fit(y[tr])
    t0 = time.perf_counter()
    m.fit(X[tr], ts.transform(y[tr]), X[va], ts.transform(y[va]))
    fit_s = time.perf_counter() - t0
    pred = ts.inverse(m.predict(X[te]).reshape(int(te.sum()), -1))
    rows = []
    subj_te = pool.w.subject[te]
    for s in np.unique(subj_te):
        ms = subj_te == s
        for j, tname in enumerate(targets):
            rows.append({"model": model, "scheme": split.scheme, "fold": split.fold, "subject": int(s),
                         "target": tname, **all_metrics(y[te][ms, j], pred[ms, j]), "n_train": int(tr.sum()),
                         "n_test": int(ms.sum()), "fit_s": fit_s, "info": getattr(m, "info", {})})
    te_idx = np.flatnonzero(te)
    preds = {"rows": te_idx, "y_true": y[te].astype(np.float32), "y_pred": pred.astype(np.float32)}
    return rows, preds
