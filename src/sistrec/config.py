"""Rutas y constantes del proyecto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

DB_PATH = DATA_RAW / "data.db"
SAMPLE_SUBMISSION = DATA_RAW / "ejemplo.csv"

USER_COL = "id_lector"
ITEM_COL = "id_libro"
DATE_COL = "fecha"
RATING_COL = "rating"

K = 20  # libros a recomendar por lector (formato de ejemplo.csv)
SEED = 42
