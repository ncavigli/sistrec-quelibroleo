"""Interfaz común para todos los recomendadores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..config import ITEM_COL, K, USER_COL
from ..data import InteractionMatrix


def interaction_weights(
    df: pd.DataFrame,
    scheme: str = "binary",
    half_life_days: float | None = None,
    ref_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """Pesos por interacción.

    scheme:
      - "binary": 1 por libro leído (lo que mide NDCG: leer o no leer).
      - "rating": rating / 10 (un 10 pesa más que un 3).
      - "rating_centered": 1 + (rating - media_usuario)/10, recortado a >= 0.1.
    half_life_days: multiplica por 0.5 ** (antigüedad / half_life): lo reciente pesa más.
    """
    if scheme == "binary":
        w = np.ones(len(df), dtype=np.float64)
    elif scheme == "rating":
        w = df["rating"].fillna(7).to_numpy(np.float64) / 10.0
    elif scheme == "rating_centered":
        r = df["rating"].fillna(7)
        mu = r.groupby(df[USER_COL]).transform("mean")
        w = np.clip(1.0 + (r - mu).to_numpy(np.float64) / 10.0, 0.1, None)
    else:
        raise ValueError(f"scheme desconocido: {scheme}")
    if half_life_days:
        ref = ref_date if ref_date is not None else df["fecha"].max()
        age = (ref - df["fecha"]).dt.days.to_numpy(np.float64)
        w = w * np.power(0.5, np.clip(age, 0, None) / half_life_days)
    return w


class Recommender(ABC):
    """Clase base.

    Las subclases implementan `_fit()` y `_score(user_idx) -> (n, n_items)`.
    La base se encarga de: codificar ids, excluir libros ya leídos, cold start y top-K.
    """

    name = "base"

    def __init__(
        self,
        weighting: str = "binary",
        half_life_days: float | None = None,
        fallback_half_life_days: float = 365.0,
    ):
        self.weighting = weighting
        self.half_life_days = half_life_days
        self.fallback_half_life_days = fallback_half_life_days

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train: pd.DataFrame,
        books: pd.DataFrame | None = None,
        extra_items: list[str] | None = None,
    ) -> "Recommender":
        """`extra_items`: libros sin interacciones que igual queremos poder recomendar."""
        self.ref_date = train["fecha"].max()
        w = interaction_weights(train, self.weighting, self.half_life_days, self.ref_date)
        self.im = InteractionMatrix.from_frame(train, weights=w, extra_items=extra_items)
        self.X = self.im.matrix  # usuarios x ítems (ponderada)
        self.seen = self.X.copy()
        self.seen.data[:] = 1.0
        self.books = books
        self.train = train
        # Popularidad reciente: fallback para lectores sin historial y para desempates.
        pw = interaction_weights(train, "binary", self.fallback_half_life_days, self.ref_date)
        pop = pd.Series(pw).groupby(train[ITEM_COL].to_numpy()).sum()
        self.popularity = pop.reindex(self.im.item_ids).fillna(0).to_numpy(np.float32)
        self._fit()
        return self

    @abstractmethod
    def _fit(self) -> None: ...

    @abstractmethod
    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        """Puntajes densos (len(user_idx), n_items). Mayor = mejor."""

    # ------------------------------------------------------------ predicción
    def score_users(self, users: Sequence[str], exclude_seen: bool = True) -> np.ndarray:
        """Matriz (len(users), n_items) de puntajes; usuarios desconocidos -> popularidad."""
        idx = np.array([self.im.user_index.get(u, -1) for u in users])
        out = np.empty((len(users), self.im.n_items), dtype=np.float32)
        known = idx >= 0
        if known.any():
            out[known] = self._score(idx[known])
        if (~known).any():
            out[~known] = self.popularity
        if exclude_seen and known.any():
            seen = self.seen[idx[known]]
            rows = np.flatnonzero(known)
            r, c = seen.nonzero()
            out[rows[r], c] = -np.inf
        return out

    def recommend(
        self,
        users: Sequence[str],
        k: int = K,
        exclude_seen: bool = True,
        batch_size: int = 512,
    ) -> dict[str, list[str]]:
        recs: dict[str, list[str]] = {}
        # desempate estable por popularidad
        tiebreak = 1e-6 * self.popularity / (self.popularity.max() + 1e-9)
        for start in range(0, len(users), batch_size):
            batch = list(users[start : start + batch_size])
            s = self.score_users(batch, exclude_seen) + tiebreak
            top = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
            order = np.take_along_axis(s, top, axis=1).argsort(axis=1)[:, ::-1]
            top = np.take_along_axis(top, order, axis=1)
            for u, row in zip(batch, top):
                recs[u] = self.im.item_ids[row].tolist()
        return recs

    def __repr__(self) -> str:
        params = {k: v for k, v in vars(self).items() if not k.startswith("_") and np.isscalar(v)}
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in params.items())})"
