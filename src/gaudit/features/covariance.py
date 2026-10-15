"""Window covariances (OAS shrinkage) and a tangent-space map with a train-only reference."""

from __future__ import annotations

import numpy as np


def oas_cov(win: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Vectorised Oracle Approximating Shrinkage covariance (Chen et al. 2010, as in sklearn).

    win: (n, C, T) -> (n, C, C) float64. CAR makes the raw covariance rank C-1; the shrinkage
    toward a scaled identity keeps every estimate positive definite.
    """
    n, C, T = win.shape
    out = np.empty((n, C, C), np.float64)
    for i in range(0, n, chunk):
        x = win[i:i + chunk].astype(np.float64)
        x = x - x.mean(-1, keepdims=True)
        S = x @ x.transpose(0, 2, 1) / T
        mu = np.trace(S, axis1=1, axis2=2) / C
        alpha = (S ** 2).mean((1, 2))
        num = alpha + mu ** 2
        den = (T + 1) * (alpha - mu ** 2 / C)
        sh = np.where(den == 0, 1.0, np.minimum(num / np.where(den == 0, 1, den), 1.0))
        out[i:i + chunk] = (1 - sh)[:, None, None] * S + (sh * mu)[:, None, None] * np.eye(C)
    return out


def mean_riemann(covs: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """Affine-invariant Riemannian (Karcher) mean by the standard fixed-point iteration
    (same algorithm as pyriemann.mean_riemann), batched with torch (GPU if available)."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = torch.as_tensor(np.asarray(covs, dtype=np.float64), device=dev)

    def fn(M, f):
        w, V = torch.linalg.eigh(0.5 * (M + M.transpose(-1, -2)))
        return (V * f(w).unsqueeze(-2)) @ V.transpose(-1, -2)

    G = C.mean(0)  # Euclidean initialisation, as in pyriemann
    for _ in range(max_iter):
        Gs, Gis = fn(G, torch.sqrt), fn(G, lambda w: 1 / torch.sqrt(w))
        T = fn(Gis @ C @ Gis, torch.log).mean(0)
        G = Gs @ fn(T, torch.exp) @ Gs
        if torch.linalg.norm(T) < tol:
            break
    return G.cpu().numpy()


def _powm(M: np.ndarray, p: float) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    return (V * w ** p) @ V.T


class TangentSpace:
    """Log-map at a Riemannian reference mean that is fit on training covariances only."""

    def __init__(self, max_ref: int = 2000, seed: int = 0):
        self.max_ref, self.seed = max_ref, seed

    def fit(self, covs: np.ndarray) -> TangentSpace:
        g = np.random.default_rng(self.seed)
        sub = covs if len(covs) <= self.max_ref else covs[g.choice(len(covs), self.max_ref, replace=False)]
        self.ref_ = mean_riemann(np.asarray(sub, dtype=np.float64))
        self.isqrt_ = _powm(self.ref_, -0.5)
        C = covs.shape[1]
        self.iu_ = np.triu_indices(C)
        self.coef_ = np.where(self.iu_[0] == self.iu_[1], 1.0, np.sqrt(2.0))
        return self

    def transform(self, covs: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Batched matrix log on the GPU when available. float32 there (relative error ~3e-5
        versus float64, checked in tests/test_covariance.py), float64 on the CPU."""
        import torch

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float32 if dev.type == "cuda" else torch.float64
        P = torch.as_tensor(self.isqrt_, dtype=dt, device=dev)
        iu0 = torch.as_tensor(self.iu_[0], device=dev)
        iu1 = torch.as_tensor(self.iu_[1], device=dev)
        coef = torch.as_tensor(self.coef_, dtype=dt, device=dev)
        out = []
        for i in range(0, len(covs), chunk):
            C = torch.as_tensor(np.asarray(covs[i:i + chunk]), device=dev).to(dt)
            M = P @ C @ P
            M = 0.5 * (M + M.transpose(1, 2))
            w, V = torch.linalg.eigh(M)
            L = (V * torch.log(torch.clamp(w, min=1e-10))[:, None, :]) @ V.transpose(1, 2)
            out.append((L[:, iu0, iu1] * coef).float().cpu().numpy())
        return np.concatenate(out)
