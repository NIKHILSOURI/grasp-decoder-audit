"""Template + EEG residual decoder: does EEG add anything beyond the cue-locked template?

Input columns: [t_rel, EEG features...]. The template is fit on the training trials, the inner
EEG decoder is trained on what the template leaves unexplained (target - template), and the
prediction is template + predicted residual. Validation residuals use the same training template.
"""

from __future__ import annotations

from .base import Decoder
from .template import TemplateDecoder


class TemplatePlus(Decoder):
    def __init__(self, inner: Decoder):
        self.inner = inner
        self.feature = "time+" + inner.feature

    def fit(self, Xtr, ytr, Xva, yva):
        self.tpl = TemplateDecoder().fit(Xtr[:, :1], ytr)
        rtr = ytr.reshape(len(ytr), -1) - self.tpl.predict(Xtr[:, :1])
        rva = yva.reshape(len(yva), -1) - self.tpl.predict(Xva[:, :1])
        self.inner.fit(Xtr[:, 1:], rtr, Xva[:, 1:], rva)
        self.info = {"inner": getattr(self.inner, "info", {})}
        return self

    def predict(self, X):
        return self.tpl.predict(X[:, :1]) + self.inner.predict(X[:, 1:]).reshape(len(X), -1)
