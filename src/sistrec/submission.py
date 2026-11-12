"""Generación y validación del CSV de submission (formato de ejemplo.csv)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .config import ITEM_COL, K, USER_COL


def to_frame(recs: Mapping[str, Sequence[str]], users: Sequence[str]) -> pd.DataFrame:
    rows = [(u, it) for u in users for it in recs[u]]
    return pd.DataFrame(rows, columns=[USER_COL, ITEM_COL])


def validate(df: pd.DataFrame, users: Sequence[str], k: int = K) -> None:
    """Falla fuerte si el archivo no cumple el formato: mejor enterarse antes de subirlo."""
    assert list(df.columns) == [USER_COL, ITEM_COL], f"columnas: {list(df.columns)}"
    assert set(df[USER_COL]) == set(users), "faltan o sobran lectores"
    per_user = df.groupby(USER_COL).size()
    assert (per_user == k).all(), f"no todos tienen {k} libros: {per_user.value_counts().to_dict()}"
    assert not df.duplicated().any(), "hay pares (lector, libro) repetidos"
    assert df[ITEM_COL].notna().all()


def write(recs: Mapping[str, Sequence[str]], users: Sequence[str], path: Path, k: int = K) -> pd.DataFrame:
    df = to_frame(recs, users)
    validate(df, users, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df
