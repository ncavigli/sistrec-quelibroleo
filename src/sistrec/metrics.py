"""Métricas de ranking top-K.

NDCG@K con relevancia binaria (el libro aparece en las interacciones ocultas del lector):
    DCG  = sum_{i=1..K} rel_i / log2(i + 1)
    IDCG = DCG del ranking ideal (todos los relevantes arriba, hasta K)
    NDCG = DCG / IDCG

Si la competencia usa relevancia graduada (rating), `ndcg_at_k` acepta un dict item->ganancia.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def ndcg_at_k(
    recommended: Sequence[str],
    relevant: Iterable[str] | Mapping[str, float],
    k: int = 20,
) -> float:
    """NDCG@k de una lista. `relevant` puede ser un conjunto (binario) o dict item->ganancia."""
    gains = dict(relevant) if isinstance(relevant, Mapping) else {r: 1.0 for r in relevant}
    if not gains:
        return 0.0
    disc = _discounts(k)
    rec = list(dict.fromkeys(recommended))[:k]  # sin duplicados
    dcg = sum(gains.get(item, 0.0) * disc[i] for i, item in enumerate(rec))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = float(np.dot(ideal, disc[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: Sequence[str], relevant: Iterable[str], k: int = 20) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(list(recommended)[:k]) & rel) / min(len(rel), k)


def evaluate(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str] | Mapping[str, float]],
    k: int = 20,
) -> dict[str, float]:
    """Promedia las métricas sobre los usuarios de `ground_truth`.

    Un usuario sin recomendaciones cuenta como 0 (igual que en Kaggle).
    """
    ndcgs, recalls, hits = [], [], []
    for user, rel in ground_truth.items():
        rec = recommendations.get(user, [])
        ndcgs.append(ndcg_at_k(rec, rel, k))
        recalls.append(recall_at_k(rec, rel, k))
        hits.append(float(len(set(list(rec)[:k]) & set(rel)) > 0))
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"recall@{k}": float(np.mean(recalls)),
        f"hitrate@{k}": float(np.mean(hits)),
        "n_users": len(ndcgs),
    }
