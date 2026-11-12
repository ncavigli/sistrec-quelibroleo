"""Entrena con TODOS los datos y genera el CSV para Kaggle.

Ejemplo:
    python scripts/make_submission.py --model rp3beta --new-items
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sistrec import submission
from sistrec.config import SUBMISSIONS
from sistrec.data import load_books, load_interactions, load_target_users, new_catalog_items
from sistrec.registry import MODELS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--new-items", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = load_interactions()
    books = load_books()
    users = load_target_users()
    ref_year = df["fecha"].max().year + 1  # datos hasta 2024 -> el test mira 2025
    extra = new_catalog_items(books, df, ref_year - 1) if args.new_items else None

    model = MODELS[args.model]().fit(df, books, extra)
    recs = model.recommend(users)
    out = Path(args.out) if args.out else SUBMISSIONS / f"{time.strftime('%Y%m%d_%H%M')}_{args.model}.csv"
    sub = submission.write(recs, users, out)
    print(f"OK {out} ({len(sub):,} filas, {sub['id_lector'].nunique()} lectores)")


if __name__ == "__main__":
    main()
