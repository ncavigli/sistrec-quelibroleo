"""Ensamble lineal de varios recomendadores ya entrenados.

Cada modelo produce puntajes en escalas distintas, así que normalizamos por usuario
(dividiendo por el máximo de la fila) antes de sumar con pesos.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .base import Recommender


class WeightedHybrid(Recommender):
    name = "hybrid"

    def __init__(self, models: Sequence[Recommender], weights: Sequence[float], **kw):
        super().__init__(**kw)
        self.models = list(models)
        self.weights = list(weights)

    def fit(self, train, books=None, extra_items=None) -> "WeightedHybrid":
        for m in self.models:
            if not hasattr(m, "im"):  # permite reutilizar modelos ya entrenados
                m.fit(train, books, extra_items)
        return super().fit(train, books, extra_items)

    def _fit(self) -> None:
        for m in self.models:
            assert np.array_equal(m.im.item_ids, self.im.item_ids), "índices de ítems distintos"

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        users = self.im.user_ids[user_idx]
        total = np.zeros((len(user_idx), self.im.n_items), dtype=np.float32)
        for m, w in zip(self.models, self.weights):
            if w == 0:
                continue
            s = m.score_users(users, exclude_seen=False)
            s = np.where(np.isfinite(s) & (s > -1e5), s, 0.0)
            mx = np.abs(s).max(axis=1, keepdims=True)
            total += w * s / np.maximum(mx, 1e-9)
        return total
