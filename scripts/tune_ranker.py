"""Compara versiones de candidatos y configuraciones del ranker sobre folds cacheados.

Cada configuración se entrena con varias semillas para separar mejoras reales del ruido.

    python scripts/tune_ranker.py --versions v1 v2 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import sistrec  # noqa: E402,F401

import numpy as np
import pandas as pd

from sistrec.config import DATA_PROCESSED, K, REPORTS
from sistrec.data import load_interactions, load_target_users
from sistrec.ranking import Fold, TwoStageRanker
from sistrec.split import temporal_split

CONFIGS = {
    "base": {},
    "small_leaves": {"num_leaves": 31, "min_child_samples": 200, "colsample_bytree": 0.5},
    "lr05_l15": {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 300},
    "deep_reg": {"num_leaves": 127, "min_child_samples": 300, "reg_lambda": 10.0,
                 "colsample_bytree": 0.5},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--versions", nargs="+", default=["v1", "v2"])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--train-folds", nargs="+", default=["2022-01-01", "2023-01-01"])
    p.add_argument("--valid-fold", default="2024-01-01")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    df = load_interactions()
    targets = load_target_users()
    gts = {c: temporal_split(df, c, None if c == args.valid_fold else
                             str((pd.Timestamp(c) + pd.DateOffset(years=1)).date())).ground_truth
           for c in args.train_folds + [args.valid_fold]}

    rows = []
    for v in args.versions:
        load = lambda c: Fold(c, pd.read_parquet(DATA_PROCESSED / f"fold_{c}_labeled_{v}.parquet"), gts[c])
        train = [load(c) for c in args.train_folds]
        valid = load(args.valid_fold)
        for cname in args.configs:
            scores, tscores, iters = [], [], []
            for seed in range(args.seeds):
                r = TwoStageRanker({**CONFIGS[cname], "random_state": seed}).fit(train, valid)
                scores.append(r.evaluate(valid)[f"ndcg@{K}"])
                tscores.append(r.evaluate(valid, subset=targets)[f"ndcg@{K}"])
                iters.append(r.best_iteration_)
            row = dict(version=v, config=cname, ndcg=np.mean(scores), ndcg_std=np.std(scores),
                       ndcg_target=np.mean(tscores), best_iter=int(np.mean(iters)))
            rows.append(row)
            print(f"{v} {cname:<13} ndcg={row['ndcg']:.4f}±{row['ndcg_std']:.4f} "
                  f"target={row['ndcg_target']:.4f} iters={row['best_iter']}", flush=True)
    out = pd.DataFrame(rows).sort_values("ndcg", ascending=False)
    out.to_csv(REPORTS / "tune_ranker.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
