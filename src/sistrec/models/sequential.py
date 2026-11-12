"""Modelos que usan el ORDEN de las lecturas.

- Sequential: transiciones "después de leer i, se lee j" (dirigidas, a corto plazo).
  Se parece a ItemKNN pero (1) es asimétrico y (2) pesa sobre todo las últimas lecturas del
  lector, que son las que mejor anticipan la próxima.
- SeriesContinuation: si el lector va por el tomo n de una saga, recomienda el n+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import ITEM_COL, USER_COL
from ..series import series_table
from .base import Recommender


class Sequential(Recommender):
    name = "sequential"

    def __init__(self, window: int = 10, max_gap_days: int = 365, last_n: int = 30,
                 recency_decay: float = 0.9, row_norm: float = 0.5, pop_beta: float = 0.25, **kw):
        super().__init__(**kw)
        self.window = window          # cuántas lecturas siguientes cuentan como "después"
        self.max_gap_days = max_gap_days
        self.last_n = last_n          # perfil del lector = sus últimas N lecturas
        self.recency_decay = recency_decay
        self.row_norm = row_norm
        self.pop_beta = pop_beta

    def _fit(self) -> None:
        t = self.train.sort_values([USER_COL, "fecha"], kind="stable")
        u = t[USER_COL].map(self.im.user_index).to_numpy()
        i = t[ITEM_COL].map(self.im.item_index).to_numpy()
        d = t["fecha"].to_numpy("datetime64[D]").astype(np.int64)
        src, dst, w = [], [], []
        for k in range(1, self.window + 1):
            gap = d[k:] - d[:-k]
            m = (u[k:] == u[:-k]) & (gap <= self.max_gap_days)
            src.append(i[:-k][m]); dst.append(i[k:][m]); w.append(np.full(m.sum(), 1.0 / k))
            same = m & (gap == 0)  # mismo día: el orden real es desconocido -> ambas direcciones
            src.append(i[k:][same]); dst.append(i[:-k][same]); w.append(np.full(same.sum(), 0.5 / k))
        n = self.im.n_items
        T = sp.csr_matrix((np.concatenate(w).astype(np.float32),
                           (np.concatenate(src), np.concatenate(dst))), shape=(n, n))
        rs = np.asarray(T.sum(axis=1)).ravel()
        pop = np.asarray(self.seen.sum(axis=0)).ravel()
        T = sp.diags(np.power(np.maximum(rs, 1e-9), -self.row_norm).astype(np.float32)) @ T
        T = T @ sp.diags(np.power(np.maximum(pop, 1.0), -self.pop_beta).astype(np.float32))
        self.T = T.tocsr()

        # perfil: últimas N lecturas con peso decay^k (k = 0 la más reciente)
        last = t.groupby(USER_COL, sort=False).tail(self.last_n)
        k = last.groupby(USER_COL, sort=False).cumcount(ascending=False).to_numpy()
        self.R = sp.csr_matrix(
            (np.power(self.recency_decay, k).astype(np.float32),
             (last[USER_COL].map(self.im.user_index).to_numpy(),
              last[ITEM_COL].map(self.im.item_index).to_numpy())),
            shape=(self.im.n_users, n),
        )

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return (self.R[user_idx] @ self.T).toarray()


class SeriesContinuation(Recommender):
    """score(u, j) = 1 / (1 + |tomo_j - (max_tomo_leído_u + 1)|) para sagas que u empezó."""

    name = "saga"

    def _fit(self) -> None:
        if self.books is None:
            raise ValueError("SeriesContinuation necesita `books`")
        st = series_table(self.books)
        st = st[st.index.isin(self.im.item_index)]
        self.item_series = st
        read = self.train[[USER_COL, ITEM_COL]].join(st, on=ITEM_COL, how="inner")
        self.user_max = read.groupby([USER_COL, "serie"])["tomo"].max()
        cols = st.index.map(self.im.item_index).to_numpy()
        self._series_items = pd.DataFrame({"serie": st["serie"].to_numpy(),
                                           "tomo": st["tomo"].to_numpy(), "col": cols})

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        out = np.full((len(user_idx), self.im.n_items), -1e6, dtype=np.float32)
        users = self.im.user_ids[user_idx]
        um = self.user_max[self.user_max.index.get_level_values(0).isin(set(users))].reset_index()
        if um.empty:
            return out
        um = um.rename(columns={"tomo": "max_tomo"})
        j = um.merge(self._series_items, on="serie")
        score = 1.0 / (1.0 + np.abs(j["tomo"] - (j["max_tomo"] + 1)))
        row = pd.Index(users).get_indexer(j[USER_COL])
        out[row, j["col"].to_numpy()] = score.to_numpy(np.float32)
        return out
