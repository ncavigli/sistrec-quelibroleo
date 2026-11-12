"""Entrena el ranker de 2 etapas, lo valida en 2024 y genera la submission.

    python scripts/train_ranker.py                         # usa folds cacheados si existen
    python scripts/train_ranker.py --rebuild               # recalcula candidatos/features
    python scripts/train_ranker.py --train-folds 2023-01-01 --no-submission
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sistrec  # noqa: E402,F401  (fija hilos de BLAS antes de numpy)

import pandas as pd

from sistrec import submission
from sistrec.config import DATA_PROCESSED, K, REPORTS, SUBMISSIONS
from sistrec.data import load_books, load_interactions, load_readers, load_target_users
from sistrec.evaluation import log_result
from sistrec.models import Popularity
from sistrec.ranking import Fold, TwoStageRanker, build_fold
from sistrec.ranking.pipeline import CANDIDATES_VERSION
from sistrec.split import temporal_split


def get_fold(df, books, readers, cutoff, end=None, users=None, rebuild=False) -> Fold:
    tag = "final" if users is not None else "labeled"
    path = DATA_PROCESSED / f"fold_{cutoff}_{tag}_{CANDIDATES_VERSION}.parquet"
    gt = temporal_split(df, cutoff, end).ground_truth if users is None else {}
    if path.exists() and not rebuild:
        frame = pd.read_parquet(path)
        print(f"[fold {cutoff}] cargado de caché: {len(frame):,} filas", flush=True)
        return Fold(cutoff, frame, gt)
    fold = build_fold(df, books, readers, cutoff, end, users=users)
    path.parent.mkdir(parents=True, exist_ok=True)
    fold.frame.to_parquet(path, index=False)
    return fold


def next_year(cutoff: str) -> str:
    return str((pd.Timestamp(cutoff) + pd.DateOffset(years=1)).date())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-folds", nargs="+",
                   default=["2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"])
    p.add_argument("--valid-fold", default="2024-01-01")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--no-submission", action="store_true")
    p.add_argument("--tag", default="ranker")
    args = p.parse_args()

    t0 = time.perf_counter()
    df, books, readers = load_interactions(), load_books(), load_readers()
    targets = load_target_users()

    train_folds = [get_fold(df, books, readers, c, next_year(c), rebuild=args.rebuild)
                   for c in args.train_folds]
    valid = get_fold(df, books, readers, args.valid_fold, None, rebuild=args.rebuild)

    # 1) Entrenar con folds pasados, early stopping en 2024
    ranker = TwoStageRanker().fit(train_folds, valid)
    res_all = ranker.evaluate(valid)
    res_tgt = ranker.evaluate(valid, subset=targets)
    print(f"\nVALIDACIÓN {args.valid_fold}: ndcg@{K}={res_all[f'ndcg@{K}']:.4f} "
          f"target={res_tgt[f'ndcg@{K}']:.4f} recall={res_all[f'recall@{K}']:.4f} "
          f"hit={res_all[f'hitrate@{K}']:.3f} best_iter={ranker.best_iteration_}")
    imp = ranker.importance()
    print("\nTop features (gain):\n" + (imp.head(20) / imp.sum()).round(3).to_string())
    REPORTS.mkdir(exist_ok=True)
    imp.to_csv(REPORTS / f"importance_{args.tag}.csv")
    log_result({"name": args.tag, "model": "TwoStageRanker", "cutoff": args.valid_fold,
                "train_folds": args.train_folds, **res_all, f"ndcg@{K}[target]": res_tgt[f"ndcg@{K}"],
                "best_iter": ranker.best_iteration_})

    if args.no_submission:
        return

    # 2) Re-entrenar sumando el fold de validación (datos más recientes = más parecidos al test)
    n_iter = int(ranker.best_iteration_ * 1.1)
    final_ranker = TwoStageRanker({"n_estimators": n_iter}).fit(train_folds + [valid])

    # 3) Fold final: todo el historial, lectores de Kaggle
    final_cutoff = str((df["fecha"].max() + pd.Timedelta(days=1)).date())
    final = get_fold(df, books, readers, final_cutoff, users=targets, rebuild=args.rebuild)
    recs = final_ranker.recommend(final.frame, K)

    # relleno por las dudas (lector con < K candidatos)
    pop = Popularity(half_life_days=90).fit(df).recommend(targets, k=K * 3)
    for u in targets:
        r = recs.get(u, [])
        recs[u] = (r + [x for x in pop[u] if x not in set(r)])[:K]

    out = SUBMISSIONS / f"{time.strftime('%Y%m%d_%H%M')}_{args.tag}.csv"
    submission.write(recs, targets, out)
    print(f"\nSubmission: {out}  ({time.perf_counter() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
