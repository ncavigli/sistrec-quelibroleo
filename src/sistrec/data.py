"""Carga y limpieza de datos desde la base SQLite provista por la cátedra.

Tablas:
  - interacciones(id_lector, id_libro, fecha 'dd-mm-YYYY', rating 1..10)
  - libros(id_libro, titulo, autor, genero, editorial, anio_edicion, isbn, resumen, img_src)
  - lectores(id_lector, nombre, genero, vive_en, nacimiento)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DB_PATH, ITEM_COL, SAMPLE_SUBMISSION, USER_COL


def _connect(db_path: Path) -> sqlite3.Connection:
    # Modo sólo lectura: nunca modificamos la base original.
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def load_interactions(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Devuelve interacciones con `fecha` como datetime, ordenadas por fecha."""
    with _connect(db_path) as con:
        df = pd.read_sql("SELECT id_lector, id_libro, fecha, rating FROM interacciones", con)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce")
    # Hay 1 fecha inválida: la mandamos al pasado para que siempre quede en train.
    df["fecha"] = df["fecha"].fillna(df["fecha"].min())
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float32")
    df = df.drop_duplicates([USER_COL, ITEM_COL], keep="last")
    return df.sort_values("fecha", kind="stable").reset_index(drop=True)


def load_books(db_path: Path = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as con:
        df = pd.read_sql(
            "SELECT id_libro, titulo, autor, genero, editorial, anio_edicion, resumen FROM libros", con
        )
    df["anio_edicion"] = pd.to_numeric(df["anio_edicion"], errors="coerce")
    for c in ("autor", "genero", "editorial"):
        df[c] = df[c].fillna("").str.strip().str.upper()
    return df.drop_duplicates(ITEM_COL).set_index(ITEM_COL)


def load_readers(db_path: Path = DB_PATH) -> pd.DataFrame:
    with _connect(db_path) as con:
        df = pd.read_sql("SELECT id_lector, genero, vive_en, nacimiento FROM lectores", con)
    df["nacimiento"] = pd.to_numeric(df["nacimiento"], errors="coerce")
    df.loc[(df["nacimiento"] < 1900) | (df["nacimiento"] > 2020), "nacimiento"] = np.nan
    return df.drop_duplicates(USER_COL).set_index(USER_COL)


def load_target_users(path: Path = SAMPLE_SUBMISSION) -> list[str]:
    """Lectores para los que hay que generar recomendaciones (en el orden del ejemplo)."""
    users = pd.read_csv(path, dtype=str)[USER_COL]
    return users.drop_duplicates().tolist()


def new_catalog_items(books: pd.DataFrame, train: pd.DataFrame, min_year: int) -> list[str]:
    """Libros del catálogo sin interacciones en `train` y editados desde `min_year`.

    Son las novedades: ningún modelo colaborativo puede recomendarlas (nadie las leyó aún),
    pero los modelos de contenido (autor, género) sí.
    """
    unseen = books.index.difference(pd.Index(train[ITEM_COL].unique()))
    yr = books.loc[unseen, "anio_edicion"]
    return yr[yr >= min_year].index.tolist()


@dataclass
class InteractionMatrix:
    """Matriz usuario x ítem dispersa + codificadores de ids.

    `values` puede ser binaria (1 = leyó el libro) o ponderada (rating, decaimiento temporal...).
    """

    matrix: sp.csr_matrix
    user_ids: np.ndarray  # índice -> id_lector
    item_ids: np.ndarray  # índice -> id_libro
    user_index: dict[str, int]
    item_index: dict[str, int]

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        weights: np.ndarray | None = None,
        extra_users: list[str] | None = None,
        extra_items: list[str] | None = None,
    ) -> "InteractionMatrix":
        user_ids = pd.Index(df[USER_COL].unique())
        if extra_users:
            user_ids = user_ids.append(pd.Index(extra_users).difference(user_ids))
        item_ids = pd.Index(df[ITEM_COL].unique())
        if extra_items:
            item_ids = item_ids.append(pd.Index(extra_items).difference(item_ids))
        rows = user_ids.get_indexer(df[USER_COL])
        cols = item_ids.get_indexer(df[ITEM_COL])
        vals = np.ones(len(df), dtype=np.float32) if weights is None else weights.astype(np.float32)
        m = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
        return cls(
            matrix=m,
            user_ids=user_ids.to_numpy(),
            item_ids=item_ids.to_numpy(),
            user_index={u: i for i, u in enumerate(user_ids)},
            item_index={it: i for i, it in enumerate(item_ids)},
        )

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]
