import numpy as np
import pandas as pd

from sistrec.models import AttributeAffinity, ItemKNN, Popularity
from sistrec.ranking import TwoStageRanker, build_fold

GENS = {
    "pop": (lambda: Popularity(half_life_days=90), 10),
    "autor": (lambda: AttributeAffinity("autor"), 10),
    "knn": (lambda: ItemKNN(k_neighbors=10), 10),
}


def _data():
    rng = np.random.default_rng(1)
    rows = []
    for u in range(80):
        for it in rng.choice(60, size=rng.integers(5, 25), replace=False):
            d = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(rng.integers(0, 1460)))
            rows.append((f"u{u}", f"b{it}", d, float(rng.integers(1, 11))))
    df = pd.DataFrame(rows, columns=["id_lector", "id_libro", "fecha", "rating"])
    books = pd.DataFrame(
        {"titulo": [f"T{i}" for i in range(65)], "autor": [f"A{i % 9}" for i in range(65)], "genero": [f"G{i % 4}" for i in range(65)],
         "editorial": ["E"] * 65, "anio_edicion": [2015.0] * 60 + [2024.0] * 5},
        index=[f"b{i}" for i in range(65)],
    )
    readers = pd.DataFrame({"genero": ["Hombre", "Mujer"] * 40, "nacimiento": [1980.0] * 80},
                           index=[f"u{i}" for i in range(80)])
    return df, books, readers


def test_two_stage_end_to_end():
    df, books, readers = _data()
    f23 = build_fold(df, books, readers, "2023-01-01", "2024-01-01", generators=GENS, verbose=False)
    f24 = build_fold(df, books, readers, "2024-01-01", generators=GENS, verbose=False)
    assert f23.frame["label"].sum() > 0
    assert not f23.frame.duplicated(["id_lector", "id_libro"]).any()
    # sin fuga: ningún candidato es un libro ya leído antes del corte
    train = df[df["fecha"] < "2023-01-01"]
    seen = set(zip(train["id_lector"], train["id_libro"]))
    assert not any((u, i) in seen for u, i in zip(f23.frame["id_lector"], f23.frame["id_libro"]))

    r = TwoStageRanker({"n_estimators": 20, "min_child_samples": 5}).fit([f23], f24)
    res = r.evaluate(f24, k=5)
    assert 0.0 <= res["ndcg@5"] <= 1.0

    final = build_fold(df, books, readers, "2025-01-01", users=["u1", "u2", "nuevo"],
                       generators=GENS, verbose=False)
    recs = r.recommend(final.frame, k=5)
    assert set(recs) == {"u1", "u2", "nuevo"}
    assert all(len(v) == 5 for v in recs.values())
