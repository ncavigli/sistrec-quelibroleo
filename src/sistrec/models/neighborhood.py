"""Modelos de vecindario ítem-ítem: ItemKNN (coseno) y RP3beta (random walk en el grafo)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


def _topk_sparse_from_blocks(block_fn, n_items: int, k: int, block: int = 512) -> sp.csr_matrix:
    """Construye una similitud ítem-ítem dispersa conservando los `k` vecinos por columna.

    `block_fn(cols)` devuelve un bloque denso (n_items, len(cols)). Procesar por bloques evita
    materializar la matriz n_items x n_items completa (48k x 48k no entra en memoria).
    """
    rows_all, cols_all, vals_all = [], [], []
    for start in range(0, n_items, block):
        cols = np.arange(start, min(start + block, n_items))
        B = block_fn(cols)  # (n_items, b)
        B[cols, np.arange(len(cols))] = 0.0  # sin auto-similitud
        kk = min(k, n_items - 1)
        top = np.argpartition(-B, kth=kk, axis=0)[:kk]  # (kk, b)
        vals = np.take_along_axis(B, top, axis=0)
        keep = vals > 0
        rows_all.append(top[keep])
        cols_all.append(np.broadcast_to(cols, top.shape)[keep])
        vals_all.append(vals[keep])
    return sp.csr_matrix(
        (np.concatenate(vals_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
        shape=(n_items, n_items),
        dtype=np.float32,
    )


class ItemKNN(Recommender):
    """Similitud coseno con shrinkage entre columnas de la matriz usuario-ítem.

    score(u, j) = sum_{i leído por u} w_ui * sim(i, j)
    """

    name = "itemknn"

    def __init__(self, k_neighbors: int = 200, shrink: float = 10.0, alpha: float = 0.5,
                 bm25: bool = False, **kw):
        super().__init__(**kw)
        self.k_neighbors = k_neighbors
        self.shrink = shrink
        self.alpha = alpha  # coseno asimétrico: 0.5 = coseno clásico
        self.bm25 = bm25

    def _fit(self) -> None:
        X = self.X.tocsc().astype(np.float32)
        if self.bm25:
            # IDF por usuario: usuarios que leen todo aportan menos señal por libro.
            n_u = np.diff(X.tocsr().indptr)
            idf = np.log(X.shape[1] / (1.0 + n_u)).astype(np.float32)
            X = sp.diags(idf) @ X
            X = X.tocsc()
        norms_sq = np.asarray(X.multiply(X).sum(axis=0)).ravel().astype(np.float32)
        XT = X.T.tocsr()

        def block_fn(cols):
            num = (XT @ X[:, cols]).toarray()
            den = np.outer(norms_sq ** self.alpha, norms_sq[cols] ** (1 - self.alpha))
            return num / (den + self.shrink + 1e-9)

        self.W = _topk_sparse_from_blocks(block_fn, X.shape[1], self.k_neighbors)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return (self.X[user_idx] @ self.W).toarray()


class RP3beta(Recommender):
    """Random walk de 3 pasos usuario->ítem->usuario->ítem, penalizando popularidad (beta).

    Paoletti et al. Muy competitivo en datasets implícitos y barato de entrenar.
    """

    name = "rp3beta"

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, k_neighbors: int = 200, **kw):
        super().__init__(**kw)
        self.alpha = alpha
        self.beta = beta
        self.k_neighbors = k_neighbors

    def _fit(self) -> None:
        X = self.X.astype(np.float32)
        # P_ui: transición usuario->ítem (filas normalizadas), P_iu: ítem->usuario
        Pui = sp.diags(1.0 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1e-9)) @ X
        Piu = sp.diags(1.0 / np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1e-9)) @ X.T.tocsr()
        Pui = Pui.power(self.alpha).tocsc()
        Piu = Piu.power(self.alpha).tocsr()
        pop = np.asarray(self.seen.sum(axis=0)).ravel()
        pen = np.power(np.maximum(pop, 1.0), -self.beta).astype(np.float32)

        def block_fn(cols):
            return (Piu @ Pui[:, cols]).toarray() * pen[cols][None, :]

        W = _topk_sparse_from_blocks(block_fn, X.shape[1], self.k_neighbors)
        # normalización por filas (recomendado en el paper)
        rs = np.asarray(W.sum(axis=1)).ravel()
        self.W = (sp.diags(1.0 / np.maximum(rs, 1e-9)) @ W).tocsr()

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return (self.X[user_idx] @ self.W).toarray()
