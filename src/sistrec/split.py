"""Split temporal para validación offline.

El dataset termina el 31-12-2024 y el test de Kaggle son lecturas que no vemos, de 832 lectores
mayoritariamente activos en 2024. Simulamos esa situación: entrenamos con todo lo anterior a
`cutoff` y usamos como verdad las lecturas en [cutoff, end) de los lectores elegidos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ITEM_COL, SEED, USER_COL


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    ground_truth: dict[str, set[str]]  # lector -> libros nuevos leídos en la ventana
    ground_truth_ratings: dict[str, dict[str, float]]
    cutoff: pd.Timestamp

    @property
    def users(self) -> list[str]:
        return list(self.ground_truth)


def temporal_split(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    n_users: int | None = None,
    min_train_interactions: int = 0,
    seed: int = SEED,
) -> TemporalSplit:
    """Corta por fecha. Sólo cuentan como relevantes los libros NO leídos antes del corte."""
    cutoff = pd.Timestamp(cutoff)
    train = df[df["fecha"] < cutoff]
    future = df[df["fecha"] >= cutoff]
    if end is not None:
        future = future[future["fecha"] < pd.Timestamp(end)]

    seen = set(zip(train[USER_COL], train[ITEM_COL]))
    mask = [(u, i) not in seen for u, i in zip(future[USER_COL], future[ITEM_COL])]
    future = future[np.asarray(mask, dtype=bool)]

    if min_train_interactions > 0:
        counts = train[USER_COL].value_counts()
        ok = set(counts[counts >= min_train_interactions].index)
        future = future[future[USER_COL].isin(ok)]

    users = future[USER_COL].unique()
    if n_users is not None and n_users < len(users):
        users = np.random.default_rng(seed).choice(users, size=n_users, replace=False)
        future = future[future[USER_COL].isin(set(users))]

    gt = future.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
    gt_r = {
        u: dict(zip(g[ITEM_COL], g["rating"])) for u, g in future.groupby(USER_COL)
    }
    return TemporalSplit(train=train.copy(), ground_truth=gt, ground_truth_ratings=gt_r, cutoff=cutoff)
