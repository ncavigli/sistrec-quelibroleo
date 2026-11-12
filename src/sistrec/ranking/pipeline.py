"""Pipeline de 2 etapas con folds temporales.

    fold 2023: generadores y features con datos < 2023-01-01, etiquetas = lecturas nuevas de 2023
    fold 2024: idem con < 2024-01-01, etiquetas = 2024           (validación del ranker)
    final:     generadores y features con TODO (< 2025), sin etiquetas -> submission

El ranker (LightGBM LambdaRank) aprende de folds pasados a combinar las señales de los
generadores con las features; como cada fold respeta el tiempo, no hay fuga de información.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import ITEM_COL, K, SEED, USER_COL
from ..data import new_catalog_items
from ..metrics import evaluate
from ..models import (ALS, EASE, AttributeAffinity, ItemKNN, NewItemAffinity, Popularity,
                      Recommender)
from ..split import temporal_split
from .candidates import candidate_recall, generate_candidates
from .features import build_features, feature_columns

# Cambiar cuando cambian generadores/features: invalida los folds cacheados.
CANDIDATES_VERSION = "v2"
NEW_ITEMS_YEARS_BACK = 3  # novedades = libros sin lecturas editados desde (año corte - 3)

# nombre -> (fábrica, cantidad de candidatos que aporta)
DEFAULT_GENERATORS: dict[str, tuple[Callable[[], Recommender], int]] = {
    "pop90": (lambda: Popularity(half_life_days=90), 100),
    "pop365": (lambda: Popularity(half_life_days=365), 50),
    "autor": (lambda: AttributeAffinity("autor", gamma=0.5, half_life_days=730), 150),
    "autor_pure": (lambda: AttributeAffinity("autor", gamma=0.0, half_life_days=730), 100),
    "novedades": (lambda: NewItemAffinity(half_life_days=730), 150),
    "genero": (lambda: AttributeAffinity("genero", gamma=1.0, half_life_days=365), 100),
    "itemknn": (lambda: ItemKNN(k_neighbors=200, shrink=10, half_life_days=730), 200),
    "ease": (lambda: EASE(l2=500, max_items=8000, half_life_days=730), 150),
    "als": (lambda: ALS(factors=128, regularization=0.05, alpha=20, iterations=20), 100),
}


@dataclass
class Fold:
    name: str
    frame: pd.DataFrame  # candidatos + features (+ label si hay verdad)
    ground_truth: dict[str, set[str]] = field(default_factory=dict)
    cand_recall: float | None = None


def build_fold(
    df: pd.DataFrame,
    books: pd.DataFrame,
    readers: pd.DataFrame | None,
    cutoff: str,
    end: str | None = None,
    users: Sequence[str] | None = None,
    generators: Mapping[str, tuple[Callable[[], Recommender], int]] = DEFAULT_GENERATORS,
    verbose: bool = True,
) -> Fold:
    t0 = time.perf_counter()
    cutoff_ts = pd.Timestamp(cutoff)
    if users is None:  # fold con etiquetas
        split = temporal_split(df, cutoff_ts, end)
        train, gt, users = split.train, split.ground_truth, split.users
    else:  # fold final (sin etiquetas)
        train, gt = df[df["fecha"] < cutoff_ts], {}
    extra = new_catalog_items(books, train, cutoff_ts.year - NEW_ITEMS_YEARS_BACK)

    fitted = {}
    for name, (factory, _) in generators.items():
        fitted[name] = factory().fit(train, books, extra)
    cands = generate_candidates(fitted, {n: k for n, (_, k) in generators.items()}, users)
    del fitted
    frame = build_features(cands, train, books, readers)
    rec = None
    if gt:
        pos = {(u, i) for u, items in gt.items() for i in items}
        frame["label"] = [int((u, i) in pos) for u, i in zip(frame[USER_COL], frame[ITEM_COL])]
        rec = candidate_recall(cands, gt)
        if verbose:  # aporte de cada generador (recall si sólo existiera ése)
            per = {n: candidate_recall(cands[cands[f"{n}_rank"].notna()], gt) for n in generators}
            print("   recall por generador: " + "  ".join(f"{n}={r:.3f}" for n, r in per.items()))
    frame = frame.sort_values(USER_COL, kind="stable").reset_index(drop=True)
    if verbose:
        msg = f"[fold {cutoff}] usuarios={len(users)} candidatos={len(frame):,} "
        msg += f"(~{len(frame) / max(len(users), 1):.0f}/usuario) extra_items={len(extra)}"
        if rec is not None:
            msg += f" recall_candidatos={rec:.3f} positivos={frame['label'].sum():,}"
        print(msg + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    return Fold(name=cutoff, frame=frame, ground_truth=gt, cand_recall=rec)


DEFAULT_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    lambdarank_truncation_level=K,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    n_estimators=2000,
    random_state=SEED,
    verbose=-1,
)


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby(USER_COL, sort=False).size().to_numpy()


def _only_users_with_positives(frame: pd.DataFrame) -> pd.DataFrame:
    has_pos = frame.groupby(USER_COL)["label"].transform("max") > 0
    return frame[has_pos]


class TwoStageRanker:
    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}

    def fit(self, train_folds: Sequence[Fold], valid_fold: Fold | None = None) -> "TwoStageRanker":
        tr = pd.concat([_only_users_with_positives(f.frame) for f in train_folds], ignore_index=True)
        tr = tr.sort_values(USER_COL, kind="stable")  # los grupos deben ser contiguos
        self.features = feature_columns(tr)
        self.model = lgb.LGBMRanker(**self.params)
        fit_kw = {}
        if valid_fold is not None:
            va = _only_users_with_positives(valid_fold.frame)
            fit_kw = dict(
                eval_X=(va[self.features],),
                eval_y=(va["label"],),
                eval_group=[_groups(va)],
                eval_at=[K],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
        self.model.fit(tr[self.features], tr["label"], group=_groups(tr), **fit_kw)
        self.best_iteration_ = self.model.best_iteration_ or self.params["n_estimators"]
        return self

    def recommend(self, frame: pd.DataFrame, k: int = K) -> dict[str, list[str]]:
        s = self.model.predict(frame[self.features], num_iteration=self.best_iteration_)
        top = (
            frame[[USER_COL, ITEM_COL]].assign(s=s)
            .sort_values([USER_COL, "s"], ascending=[True, False])
            .groupby(USER_COL, sort=False).head(k)
        )
        return top.groupby(USER_COL, sort=False)[ITEM_COL].agg(list).to_dict()

    def evaluate(self, fold: Fold, k: int = K, subset: Sequence[str] | None = None) -> dict:
        recs = self.recommend(fold.frame, k)
        gt = fold.ground_truth
        if subset is not None:
            gt = {u: gt[u] for u in subset if u in gt}
        return evaluate(recs, gt, k)

    def importance(self) -> pd.Series:
        imp = self.model.booster_.feature_importance("gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)
