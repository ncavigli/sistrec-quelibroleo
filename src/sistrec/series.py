"""Detección de sagas/series a partir del título.

Formatos frecuentes en quelibroleo:
    "ANSIA (Serie Crave 3)"                       -> serie "crave", tomo 3
    "LA NOCHE DEL MAL (Trilogía Sol negro 2)"     -> serie "sol negro", tomo 2
    "OFRENDA A LA TORMENTA. Trilogía del Baztán 3" -> serie "del baztan", tomo 3
    "EL LIBRO DE NOAH: SAGA VANIR VIII"           -> serie "vanir", tomo 8

Descarta rangos de años "(1864-1910)" y paréntesis sin nombre "(7)".
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
          "X": 10, "XI": 11, "XII": 12}
_NUM = r"(?:[nN][ºo°.]?\s*)?(\d{1,2}|[IVX]{1,4})"
_PATTERNS = [
    re.compile(r"\(\s*([^()]*?[a-zA-ZÀ-ÿ][^()]*?)\s+" + _NUM + r"\s*\)\s*\.?\s*$"),
    re.compile(r"[.:]\s*([^.:()]*?[a-zA-ZÀ-ÿ][^.:()]*?)\s+" + _NUM + r"\s*$"),
]
_PREFIX = re.compile(r"^(serie|saga|trilogia|tetralogia|pentalogia|ciclo|cronicas?)\s+(de\s+(la|los|las|el)\s+|del\s+|de\s+)?")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _PREFIX.sub("", s).strip()


def _vol(tok: str) -> float:
    tok = tok.upper()
    if tok.isdigit():
        return float(int(tok))
    return float(_ROMAN.get(tok, np.nan))


def parse_series(titles: pd.Series) -> pd.DataFrame:
    """Devuelve DataFrame (serie, tomo) indexado como `titles`; NaN si no es parte de una saga."""
    names, vols = [], []
    for t in titles.fillna("").astype(str):
        name = vol = None
        for pat in _PATTERNS:
            m = pat.search(t.strip())
            if m:
                cand = _norm(m.group(1))
                v = _vol(m.group(2))
                if len(cand) >= 3 and not np.isnan(v) and 0 < v <= 30:
                    name, vol = cand, v
                    break
        names.append(name)
        vols.append(vol)
    return pd.DataFrame({"serie": names, "tomo": vols}, index=titles.index)


def series_table(books: pd.DataFrame) -> pd.DataFrame:
    """Serie + tomo por libro. La serie se identifica por (nombre normalizado, autor) para no
    mezclar sagas homónimas de autores distintos."""
    s = parse_series(books["titulo"])
    ok = s["serie"].notna()
    s.loc[ok, "serie"] = s.loc[ok, "serie"] + " | " + books.loc[ok, "autor"].fillna("")
    return s[ok]
