"""Corre experimentos sobre un split temporal y registra resultados."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import K, REPORTS
from .metrics import evaluate
from .models.base import Recommender
from .split import TemporalSplit


def run_experiment(
    model: Recommender,
    split: TemporalSplit,
    books: pd.DataFrame | None = None,
    extra_items: list[str] | None = None,
    k: int = K,
    subsets: dict[str, list[str]] | None = None,
) -> dict:
    t0 = time.perf_counter()
    if not hasattr(model, "im"):
        model.fit(split.train, books, extra_items)
    t_fit = time.perf_counter() - t0
    recs = model.recommend(split.users, k=k)
    res = {"model": repr(model), **evaluate(recs, split.ground_truth, k)}
    for name, users in (subsets or {}).items():
        gt = {u: split.ground_truth[u] for u in users if u in split.ground_truth}
        res[f"ndcg@{k}[{name}]"] = evaluate(recs, gt, k)[f"ndcg@{k}"]
    res["fit_s"] = round(t_fit, 1)
    res["total_s"] = round(time.perf_counter() - t0, 1)
    return res


def log_result(res: dict, path: Path = REPORTS / "experiments.jsonl") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**res, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
