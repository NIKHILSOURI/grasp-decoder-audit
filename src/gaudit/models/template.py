"""No-EEG baseline: the mean cue-locked target curve of the training trials.

The only input is the time of the window end since the segment start (segments start
0.8 s before LED-on, so this is time since the cue). It shows how much of a decoder's score
can come from the stereotyped, cue-locked shape of the movement without any brain signal.
"""

from __future__ import annotations

import numpy as np

from .base import Decoder


class TemplateDecoder(Decoder):
    feature = "time"

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        b = np.round(Xtr[:, 0] * 100).astype(int)
        y2 = ytr.reshape(len(ytr), -1)
        self.nb_ = b.max() + 1
        S = np.zeros((self.nb_, y2.shape[1]))
        n = np.bincount(b, minlength=self.nb_).astype(float)
        for j in range(y2.shape[1]):
            S[:, j] = np.bincount(b, weights=y2[:, j], minlength=self.nb_)
        T = np.full_like(S, np.nan)
        T[n > 0] = S[n > 0] / n[n > 0, None]
        # bins never seen in training: carry the last seen value forward
        for j in range(T.shape[1]):
            col = T[:, j]
            idx = np.where(~np.isnan(col), np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[:] = col[idx]
            col[np.isnan(col)] = np.nanmean(col)
        self.T_ = T
        self.info = {"bins": int(self.nb_)}
        return self

    def predict(self, X):
        b = np.clip(np.round(X[:, 0] * 100).astype(int), 0, self.nb_ - 1)
        return self.T_[b]
