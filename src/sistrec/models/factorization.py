"""Factorización matricial para feedback implícito (Hu, Koren & Volinsky 2008) vía `implicit`."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # recomendado por implicit

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from ..config import SEED
from .base import Recommender


class ALS(Recommender):
    name = "als"

    def __init__(self, factors: int = 128, regularization: float = 0.05, alpha: float = 20.0,
                 iterations: int = 20, **kw):
        super().__init__(**kw)
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations

    def _fit(self) -> None:
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=SEED,
            use_gpu=False,
        )
        self.model.fit(self.X.tocsr().astype(np.float32), show_progress=False)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.model.user_factors[user_idx] @ self.model.item_factors.T


class BPR(Recommender):
    name = "bpr"

    def __init__(self, factors: int = 128, learning_rate: float = 0.01,
                 regularization: float = 0.01, iterations: int = 100, **kw):
        super().__init__(**kw)
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations

    def _fit(self) -> None:
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=SEED,
            use_gpu=False,
        )
        self.model.fit(self.X.tocsr().astype(np.float32), show_progress=False)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.model.user_factors[user_idx] @ self.model.item_factors.T
