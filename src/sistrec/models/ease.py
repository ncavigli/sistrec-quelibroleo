"""EASE^R (Steck, 2019): autoencoder lineal con solución cerrada.

B = I - P / diag(P),  P = (X^T X + lambda I)^-1,  diag(B) = 0

Necesita invertir una matriz densa n_items x n_items, así que restringimos a los `max_items`
libros más leídos (los demás casi nunca entran en un top-20 de todas formas).
"""

from __future__ import annotations

import numpy as np

from .base import Recommender


class EASE(Recommender):
    name = "ease"

    def __init__(self, l2: float = 500.0, max_items: int = 8000, **kw):
        super().__init__(**kw)
        self.l2 = l2
        self.max_items = max_items

    def _fit(self) -> None:
        pop = np.asarray(self.seen.sum(axis=0)).ravel()
        self.items = np.sort(np.argsort(-pop)[: self.max_items])
        Xs = self.X[:, self.items].astype(np.float32)
        G = (Xs.T @ Xs).toarray()
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))[None, :]
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        out = np.full((len(user_idx), self.im.n_items), -1e6, dtype=np.float32)
        out[:, self.items] = (self.X[user_idx][:, self.items] @ self.B)
        return out
