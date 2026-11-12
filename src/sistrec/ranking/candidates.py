"""Etapa 1: generación de candidatos.

Cada generador (un Recommender ya entrenado) aporta sus top-N libros por lector. Unimos todo y,
para cada par (lector, libro) candidato, guardamos el puntaje y la posición que le dio *cada*
generador: esas columnas son features para el ranker de la etapa 2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import ITEM_COL, USER_COL
from ..models.base import Recommender


def generate_candidates(
    generators: Mapping[str, Recommender],
    n_per_generator: Mapping[str, int],
    users: Sequence[str],
    batch_size: int = 256,
) -> pd.DataFrame:
    """Devuelve un DataFrame (id_lector, id_libro, <gen>_score, <gen>_rank) por candidato.

    Todos los generadores deben haberse entrenado con el mismo train/extra_items para compartir
    el índice de ítems.
    """
    names = list(generators)
    ref = generators[names[0]].im.item_ids
    for n in names[1:]:
        assert np.array_equal(generators[n].im.item_ids, ref), f"{n}: índice de ítems distinto"

    users = list(users)
    frames = []
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores, ranks = {}, {}
        cand_mask = np.zeros((len(batch), len(ref)), dtype=bool)
        for name in names:
            s = generators[name].score_users(batch, exclude_seen=True)
            s = np.where(np.isfinite(s), s, np.nan).astype(np.float32)
            n = n_per_generator[name]
            filled = np.nan_to_num(s, nan=-np.inf)
            top = np.argpartition(-filled, kth=n, axis=1)[:, :n]
            order = np.argsort(-np.take_along_axis(filled, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            valid = np.isfinite(np.take_along_axis(filled, top, axis=1)) & (
                np.take_along_axis(filled, top, axis=1) > -1e5
            )
            rk = np.full(s.shape, np.nan, dtype=np.float32)
            rows = np.repeat(np.arange(len(batch)), n).reshape(len(batch), n)
            rk[rows[valid], top[valid]] = np.broadcast_to(np.arange(1, n + 1), top.shape)[valid]
            cand_mask[rows[valid], top[valid]] = True
            # puntaje normalizado por el máximo del lector (escalas comparables entre lectores)
            mx = np.nanmax(np.where(s > -1e5, s, np.nan), axis=1, keepdims=True)
            scores[name] = s / np.where(np.abs(mx) > 0, np.abs(mx), 1.0)
            ranks[name] = rk
        r, c = np.nonzero(cand_mask)
        df = pd.DataFrame({USER_COL: np.asarray(batch, dtype=object)[r], ITEM_COL: ref[c]})
        for name in names:
            df[f"{name}_score"] = scores[name][r, c]
            df[f"{name}_rank"] = ranks[name][r, c]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["n_generators"] = out[[f"{n}_rank" for n in names]].notna().sum(axis=1).astype(np.int8)
    return out


def candidate_recall(cands: pd.DataFrame, ground_truth: Mapping[str, set[str]]) -> float:
    """Fracción de los libros relevantes que quedaron entre los candidatos (techo del ranker)."""
    got = cands.groupby(USER_COL)[ITEM_COL].agg(set)
    hits = total = 0
    for u, rel in ground_truth.items():
        hits += len(rel & got.get(u, set()))
        total += len(rel)
    return hits / max(total, 1)
