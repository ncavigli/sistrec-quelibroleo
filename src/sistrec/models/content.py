"""Modelos basados en contenido: afinidad por autor y por género.

Intuición: quien leyó 5 libros de Pérez-Reverte probablemente lea el sexto. El colaborativo puro
capta esto sólo si otros lectores ya conectaron esos libros; el contenido lo capta directamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender

_EMPTY = {"", "-", "VARIOS", "VARIOS AUTORES", "AA.VV.", "AA. VV.", "ANÓNIMO", "ANONIMO"}


class AttributeAffinity(Recommender):
    """score(u, j) = sum_a afinidad(u, a) * [j tiene atributo a] * (pop_j normalizada)^gamma.

    afinidad(u, a) = peso de los libros leídos por u con atributo a (con decaimiento temporal
    si `half_life_days`). `gamma` mezcla con popularidad dentro del atributo.
    """

    name = "attr"

    def __init__(self, attribute: str = "autor", gamma: float = 0.5, **kw):
        super().__init__(**kw)
        self.attribute = attribute
        self.gamma = gamma

    def _fit(self) -> None:
        if self.books is None:
            raise ValueError("AttributeAffinity necesita `books` en fit()")
        attr = (
            self.books[self.attribute].reindex(self.im.item_ids).fillna("").astype(str).to_numpy()
        )
        valid = ~pd.Series(attr).isin(_EMPTY).to_numpy()
        codes, uniques = pd.factorize(attr)
        rows = np.flatnonzero(valid)
        # ítem x atributo (one-hot)
        self.A = sp.csr_matrix(
            (np.ones(rows.size, np.float32), (rows, codes[rows])),
            shape=(self.im.n_items, len(uniques)),
        )
        pop = self.popularity / (self.popularity.max() + 1e-9)
        self._item_boost = np.power(pop + 1e-3, self.gamma).astype(np.float32)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        ua = self.X[user_idx] @ self.A  # usuario x atributo
        s = (ua @ self.A.T).toarray()
        return s * self._item_boost[None, :]
