"""Baseline: los libros más leídos (opcionalmente con más peso a lo reciente)."""

from __future__ import annotations

import numpy as np

from .base import Recommender


class Popularity(Recommender):
    name = "popularity"

    def _fit(self) -> None:
        self._pop = np.asarray(self.X.sum(axis=0)).ravel().astype(np.float32)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._pop, (len(user_idx), self._pop.size)).copy()
