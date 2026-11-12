"""Catálogo de configuraciones de modelos por nombre (para scripts y experimentos)."""

from __future__ import annotations

from collections.abc import Callable

from .models import (ALS, EASE, AttributeAffinity, ItemKNN, Popularity, Recommender, RP3beta,
                     Sequential, SeriesContinuation)

MODELS: dict[str, Callable[[], Recommender]] = {
    "pop": lambda: Popularity(),
    "pop_recent": lambda: Popularity(half_life_days=180),
    "itemknn": lambda: ItemKNN(k_neighbors=200, shrink=10),
    "itemknn_decay": lambda: ItemKNN(k_neighbors=200, shrink=10, half_life_days=730),
    "rp3beta": lambda: RP3beta(alpha=1.0, beta=0.5, k_neighbors=200),
    "rp3beta_decay": lambda: RP3beta(alpha=1.0, beta=0.5, k_neighbors=200, half_life_days=730),
    "ease": lambda: EASE(l2=500, max_items=8000),
    "als": lambda: ALS(factors=128, regularization=0.05, alpha=20, iterations=20),
    "autor": lambda: AttributeAffinity("autor", gamma=0.5, half_life_days=730),
    "sequential": lambda: Sequential(),
    "sequential_w20": lambda: Sequential(window=20, last_n=50),
    "sequential_nopen": lambda: Sequential(pop_beta=0.0),
    "saga": lambda: SeriesContinuation(),
    "genero": lambda: AttributeAffinity("genero", gamma=1.0, half_life_days=365),
}
