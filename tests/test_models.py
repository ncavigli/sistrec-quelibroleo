"""Tests de humo sobre un dataset sintético chico: cada modelo entrena, excluye lo visto y
devuelve K ítems por usuario."""

import numpy as np
import pandas as pd
import pytest

from sistrec.models import (EASE, AttributeAffinity, ItemKNN, NewItemAffinity, Popularity, RP3beta,
                            WeightedHybrid)
from sistrec.split import temporal_split


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(60):
        for it in rng.choice(40, size=rng.integers(3, 15), replace=False):
            rows.append((f"u{u}", f"b{it}", pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 700))), float(rng.integers(1, 11))))
    df = pd.DataFrame(rows, columns=["id_lector", "id_libro", "fecha", "rating"])
    books = pd.DataFrame(
        {"autor": [f"A{i % 7}" for i in range(45)], "genero": ["G"] * 45,
         "editorial": ["E"] * 45, "anio_edicion": [2020] * 40 + [2025] * 5},
        index=[f"b{i}" for i in range(45)],
    )
    return df, books


MODELS = [
    lambda: Popularity(),
    lambda: ItemKNN(k_neighbors=10),
    lambda: RP3beta(k_neighbors=10),
    lambda: EASE(l2=10, max_items=30),
    lambda: AttributeAffinity("autor"),
    lambda: NewItemAffinity(),
    lambda: WeightedHybrid([ItemKNN(k_neighbors=10), AttributeAffinity("autor")], [1.0, 0.5]),
]


@pytest.mark.parametrize("factory", MODELS)
def test_model_recommends_unseen(data, factory):
    df, books = data
    sp_ = temporal_split(df, "2024-06-01")
    extra = ["b40", "b41"]
    m = factory().fit(sp_.train, books, extra_items=extra)
    users = sp_.users[:10] + ["usuario_nuevo"]
    recs = m.recommend(users, k=5)
    seen = sp_.train.groupby("id_lector")["id_libro"].agg(set)
    for u in users:
        assert len(recs[u]) == 5
        assert len(set(recs[u])) == 5
        assert not (set(recs[u]) & seen.get(u, set()))


def test_split_excludes_seen(data):
    df, _ = data
    s = temporal_split(df, "2024-06-01")
    seen = s.train.groupby("id_lector")["id_libro"].agg(set)
    for u, items in s.ground_truth.items():
        assert not (items & seen.get(u, set()))
