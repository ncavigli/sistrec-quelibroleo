"""Modelos basados en contenido: afinidad por autor y por género.

Intuición: quien leyó 5 libros de Pérez-Reverte probablemente lea el sexto. El colaborativo puro
capta esto sólo si otros lectores ya conectaron esos libros; el contenido lo capta directamente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender

_EMPTY = {"", "-", "VARIOS", "VARIOS AUTORES", "AA VV", "AAVV", "ANONIMO", "DESCONOCIDO"}


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


def _onehot(values: np.ndarray) -> sp.csr_matrix:
    """Matriz ítem x atributo (one-hot), ignorando valores vacíos/genéricos."""
    valid = ~pd.Series(values).isin(_EMPTY).to_numpy()
    codes, uniques = pd.factorize(values)
    rows = np.flatnonzero(valid)
    return sp.csr_matrix(
        (np.ones(rows.size, np.float32), (rows, codes[rows])), shape=(len(values), len(uniques))
    )


class NewItemAffinity(Recommender):
    """Sólo puntúa libros SIN lecturas en train (novedades del catálogo).

    score(u, j) = w_autor * share_u(autor_j) + w_genero * share_u(genero_j)
                  + w_editorial * share_u(editorial_j) + w_prior * popularidad_reciente(autor_j)
                  + w_alta * exp(-días desde el alta / 180)

    share_u(a) = fracción (con decaimiento temporal) de las lecturas de u con atributo a.
    El prior de autor ayuda a decidir entre novedades: un libro nuevo de un autor muy leído
    últimamente probablemente se lea mucho.
    """

    name = "novedades"

    def __init__(self, w_autor: float = 1.0, w_genero: float = 0.3, w_editorial: float = 0.1,
                 w_prior: float = 0.2, w_alta: float = 0.0, **kw):
        super().__init__(**kw)
        self.w_autor, self.w_genero = w_autor, w_genero
        self.w_editorial, self.w_prior = w_editorial, w_prior
        self.w_alta = w_alta

    def _fit(self) -> None:
        if self.books is None:
            raise ValueError("NewItemAffinity necesita `books` en fit()")
        counts = np.asarray(self.seen.sum(axis=0)).ravel()
        self.new_idx = np.flatnonzero(counts == 0)
        meta = self.books.reindex(self.im.item_ids)
        rs = np.asarray(self.X.sum(axis=1)).ravel()
        self._inv_rowsum = (1.0 / np.maximum(rs, 1e-9)).astype(np.float32)
        self.parts = []
        for attr, w in (("autor", self.w_autor), ("genero", self.w_genero),
                        ("editorial", self.w_editorial)):
            A = _onehot(meta[attr].fillna("").astype(str).to_numpy())
            self.parts.append((w, A.tocsc(), A[self.new_idx].T.tocsr()))
        # prior: popularidad reciente (decaída) del autor, normalizada a [0, 1]
        A_aut = self.parts[0][1]
        aut_pop = A_aut.T @ self.popularity
        prior = A_aut[self.new_idx] @ aut_pop
        self.prior = (prior / (prior.max() + 1e-9)).astype(np.float32)
        # recencia del alta en el sitio (dado de alta después del corte = 1; desconocido = 0)
        self.alta_rec = np.zeros(self.new_idx.size, dtype=np.float32)
        if self.w_alta and "fecha_alta" in meta:
            alta = pd.DatetimeIndex(meta["fecha_alta"].to_numpy()[self.new_idx])
            days = np.clip((self.ref_date - alta).days.to_numpy(float), 0, None)
            self.alta_rec = np.nan_to_num(np.exp(-days / 180.0)).astype(np.float32)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        out = np.full((len(user_idx), self.im.n_items), -1e6, dtype=np.float32)
        if self.new_idx.size == 0:
            return out
        P = sp.diags(self._inv_rowsum[user_idx]) @ self.X[user_idx]  # perfil normalizado
        s = np.zeros((len(user_idx), self.new_idx.size), dtype=np.float32)
        for w, A, A_new_T in self.parts:
            if w:
                s += w * ((P @ A) @ A_new_T).toarray()
        s += self.w_prior * self.prior[None, :] + self.w_alta * self.alta_rec[None, :]
        out[:, self.new_idx] = s
        return out
