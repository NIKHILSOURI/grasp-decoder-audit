"""EEGNet (braindecode) with a linear regression head on raw 1-40 Hz windows (32 x 50)."""

from __future__ import annotations

import warnings

import numpy as np

from .base import Decoder
from .train_torch import fit_regressor, predict


class EEGNetDecoder(Decoder):
    feature = "raw"

    def __init__(self, max_epochs=40, patience=8, batch=256, lr=1e-3, seed=0):
        self.kw = dict(max_epochs=max_epochs, patience=patience, batch=batch, lr=lr, seed=seed,
                       weight_decay=1e-2)
        self.seed = seed

    def fit(self, Xtr, ytr, Xva, yva):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from braindecode.models import EEGNet
        import torch

        torch.manual_seed(self.seed)
        self.scale_ = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6  # per channel, training windows only
        y2 = ytr.reshape(len(ytr), -1)
        net = EEGNet(n_chans=Xtr.shape[1], n_outputs=y2.shape[1], n_times=Xtr.shape[2], sfreq=100)
        self.net, self.info = fit_regressor(net, Xtr / self.scale_, y2, Xva / self.scale_, yva, **self.kw)
        self.info.pop("val_curve", None)
        return self

    def predict(self, X):
        return predict(self.net, (X / self.scale_).astype(np.float32))
