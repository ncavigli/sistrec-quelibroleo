"""Valida modelos con un split temporal y guarda resultados en reports/experiments.jsonl.

Ejemplo:
    python scripts/validate.py --cutoff 2024-01-01 --models pop pop_recent itemknn rp3beta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sistrec.data import load_books, load_interactions, load_target_users, new_catalog_items
from sistrec.evaluation import log_result, run_experiment
from sistrec.registry import MODELS
from sistrec.split import temporal_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cutoff", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--new-items", action="store_true", help="sumar novedades del catálogo")
    args = p.parse_args()

    df = load_interactions()
    books = load_books()
    targets = set(load_target_users())
    split = temporal_split(df, args.cutoff, args.end)
    cutoff_year = pd.Timestamp(args.cutoff).year
    extra = new_catalog_items(books, split.train, cutoff_year - 1) if args.new_items else None
    subsets = {"target": [u for u in split.users if u in targets]}
    print(f"split {args.cutoff}: train={len(split.train):,} val_users={len(split.users)} "
          f"(target={len(subsets['target'])}) extra_items={len(extra or [])}")

    rows = []
    for name in args.models:
        model = MODELS[name]()
        res = run_experiment(model, split, books, extra, subsets=subsets)
        res.update(name=name, cutoff=args.cutoff, new_items=args.new_items)
        log_result(res)
        rows.append(res)
        print(f"{name:<18} ndcg@20={res['ndcg@20']:.4f}  target={res['ndcg@20[target]']:.4f}  "
              f"recall@20={res['recall@20']:.4f}  hit={res['hitrate@20']:.3f}  ({res['total_s']}s)",
              flush=True)
        del model

    print(pd.DataFrame(rows)[["name", "ndcg@20", "ndcg@20[target]", "recall@20", "hitrate@20", "total_s"]]
          .sort_values("ndcg@20", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
