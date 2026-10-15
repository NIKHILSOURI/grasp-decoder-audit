"""Shared PyTorch training loop: MSE, AdamW, early stopping on the validation set, AMP."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(n: int, bs: int, g: torch.Generator | None):
    order = torch.randperm(n, generator=g) if g is not None else torch.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def _to_tensor(a: np.ndarray, on_gpu: bool):
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return t.to(DEVICE) if on_gpu else t.pin_memory() if DEVICE.type == "cuda" else t


@torch.no_grad()
def predict(net: nn.Module, X: np.ndarray, bs: int = 4096) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + bs], dtype=np.float32)).to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            out.append(net(xb).float().cpu().numpy())
    return np.concatenate(out).reshape(len(X), -1)


def fit_regressor(net: nn.Module, Xtr, ytr, Xva, yva, *, lr=1e-3, weight_decay=1e-2, batch=256,
                  max_epochs=40, patience=8, seed=0, amp=True, log=None) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    net = net.to(DEVICE)
    on_gpu = DEVICE.type == "cuda" and Xtr.nbytes < 2.5e9
    Xt, yt = _to_tensor(Xtr, on_gpu), _to_tensor(ytr.reshape(len(ytr), -1), on_gpu)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp and DEVICE.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, hist = np.inf, None, 0, []
    yva2 = yva.reshape(len(yva), -1)
    for _ in range(max_epochs):
        net.train()
        for idx in _batches(len(Xt), batch, g):
            idx_d = idx.to(Xt.device)
            xb = Xt[idx_d].to(DEVICE, non_blocking=True)
            yb = yt[idx_d].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=amp and DEVICE.type == "cuda"):
                loss = nn.functional.mse_loss(net(xb).float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        val = float(((predict(net, Xva) - yva2) ** 2).mean())
        hist.append(val)
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, {"epochs": len(hist), "best_val_mse": best, "val_curve": hist}
