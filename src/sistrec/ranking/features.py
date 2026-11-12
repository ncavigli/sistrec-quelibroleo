"""Etapa 2: features para cada par (lector, libro) candidato.

Todas se calculan SÓLO con el `train` del fold (fecha < cutoff) para no filtrar el futuro.

Grupos:
  - Libro:          popularidad en ventanas recientes, tendencia, rating medio, antigüedad, novedad.
  - Autor:          popularidad reciente del autor.
  - Lector:         actividad, recencia, rating medio, gusto por novedades.
  - Lector x libro: cuántos libros leyó del autor/género/editorial, cuándo fue la última vez,
                    qué rating le dio al autor, distancia al año de edición que suele leer.
  - Sagas:          tomo del libro, cuántos tomos de la saga leyó, si es el siguiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ITEM_COL, USER_COL
from ..series import series_table

WINDOWS = (30, 90, 180, 365)


def _item_features(train: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    age = (ref - train["fecha"]).dt.days
    g = train.groupby(ITEM_COL)
    f = pd.DataFrame({
        "item_cnt_all": g.size(),
        "item_rating_mean": g["rating"].mean(),
        "item_days_since_first": (ref - g["fecha"].min()).dt.days,
        "item_days_since_last": (ref - g["fecha"].max()).dt.days,
    })
    for w in WINDOWS:
        f[f"item_cnt_{w}d"] = train[age < w].groupby(ITEM_COL).size()
    f = f.fillna({f"item_cnt_{w}d": 0 for w in WINDOWS})
    f["item_trend_90_365"] = (f["item_cnt_90d"] + 1) / (f["item_cnt_365d"] / 4 + 1)
    f["item_rating_n_adj"] = (f["item_rating_mean"] * f["item_cnt_all"] + 7 * 10) / (f["item_cnt_all"] + 10)
    return f


def _user_features(train: pd.DataFrame, books: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    age = (ref - train["fecha"]).dt.days
    g = train.groupby(USER_COL)
    yr = books["anio_edicion"].reindex(train[ITEM_COL]).to_numpy()
    read_year = train["fecha"].dt.year.to_numpy()
    is_new = np.where(np.isnan(yr), np.nan, ((read_year - yr) <= 1).astype(float))
    tmp = pd.DataFrame({USER_COL: train[USER_COL].to_numpy(), "ed": yr, "is_new_rel": is_new})
    f = pd.DataFrame({
        "user_cnt_all": g.size(),
        "user_rating_mean": g["rating"].mean(),
        "user_rating_std": g["rating"].std(),
        "user_days_since_last": (ref - g["fecha"].max()).dt.days,
        "user_days_since_first": (ref - g["fecha"].min()).dt.days,
        "user_ed_year_mean": tmp.groupby(USER_COL)["ed"].mean(),
        "user_frac_new_releases": tmp.groupby(USER_COL)["is_new_rel"].mean(),
    })
    for w in (90, 365):
        f[f"user_cnt_{w}d"] = train[age < w].groupby(USER_COL).size()
    return f.fillna({"user_cnt_90d": 0, "user_cnt_365d": 0})


def _user_attr_features(train_b: pd.DataFrame, attr: str, ref: pd.Timestamp) -> pd.DataFrame:
    """Afinidad del lector con un atributo (autor/genero/editorial)."""
    d = train_b[train_b[attr] != ""]
    g = d.groupby([USER_COL, attr])
    f = pd.DataFrame({
        f"ua_{attr}_cnt": g.size(),
        f"ua_{attr}_rating": g["rating"].mean(),
        f"ua_{attr}_days_since": (ref - g["fecha"].max()).dt.days,
    })
    tot = train_b.groupby(USER_COL).size()
    f[f"ua_{attr}_share"] = f[f"ua_{attr}_cnt"] / tot.reindex(f.index.get_level_values(0)).to_numpy()
    return f


def build_features(
    cands: pd.DataFrame,
    train: pd.DataFrame,
    books: pd.DataFrame,
    readers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ref = train["fecha"].max() + pd.Timedelta(days=1)
    meta = books[["autor", "genero", "editorial", "anio_edicion"]]
    df = cands.join(meta, on=ITEM_COL)
    for c in ("autor", "genero", "editorial"):
        df[c] = df[c].fillna("")

    # --- libro
    df = df.join(_item_features(train, ref), on=ITEM_COL)
    df["item_is_new_catalog"] = df["item_cnt_all"].isna().astype(np.int8)
    df[["item_cnt_all"] + [f"item_cnt_{w}d" for w in WINDOWS]] = df[
        ["item_cnt_all"] + [f"item_cnt_{w}d" for w in WINDOWS]
    ].fillna(0)
    df["item_ed_age"] = ref.year - df["anio_edicion"]

    # --- autor (popularidad del autor en el último año y total)
    train_b = train.join(meta, on=ITEM_COL)
    for c in ("autor", "genero", "editorial"):
        train_b[c] = train_b[c].fillna("")
    recent = train_b[(ref - train_b["fecha"]).dt.days < 365]
    a_recent = recent[recent["autor"] != ""].groupby("autor").size().rename("autor_cnt_365d")
    a_all = train_b[train_b["autor"] != ""].groupby("autor").size().rename("autor_cnt_all")
    df = df.join(a_recent, on="autor").join(a_all, on="autor")
    df[["autor_cnt_365d", "autor_cnt_all"]] = df[["autor_cnt_365d", "autor_cnt_all"]].fillna(0)

    # --- lector
    df = df.join(_user_features(train, books, ref), on=USER_COL)

    # --- lector x atributo
    for attr in ("autor", "genero", "editorial"):
        uf = _user_attr_features(train_b, attr, ref)
        df = df.join(uf, on=[USER_COL, attr])
        df[f"ua_{attr}_cnt"] = df[f"ua_{attr}_cnt"].fillna(0)
        df[f"ua_{attr}_share"] = df[f"ua_{attr}_share"].fillna(0)

    # --- sagas: ¿el lector viene leyendo esta saga? ¿éste es el tomo siguiente?
    st = series_table(books)
    df = df.join(st.rename(columns={"serie": "_serie", "tomo": "item_saga_tomo"}), on=ITEM_COL)
    read = train[[USER_COL, ITEM_COL, "fecha"]].join(st, on=ITEM_COL, how="inner")
    g = read.groupby([USER_COL, "serie"])
    us = pd.DataFrame({
        "ua_saga_cnt": g.size(),
        "ua_saga_max_tomo": g["tomo"].max(),
        "ua_saga_days_since": (ref - g["fecha"].max()).dt.days,
    })
    us.index = us.index.set_names([USER_COL, "_serie"])
    df = df.join(us, on=[USER_COL, "_serie"])
    saga_cols = ["item_saga_tomo", "ua_saga_cnt", "ua_saga_max_tomo", "ua_saga_days_since"]
    df[saga_cols] = df[saga_cols].astype("float32")
    df["ua_saga_cnt"] = df["ua_saga_cnt"].fillna(0)
    df["ua_saga_next_gap"] = df["item_saga_tomo"] - df["ua_saga_max_tomo"]  # 1 = tomo siguiente
    df = df.drop(columns=["_serie"])

    df["ed_year_gap"] = df["anio_edicion"] - df["user_ed_year_mean"]
    df["ua_autor_rating_rel"] = df["ua_autor_rating"] - df["user_rating_mean"]

    # --- demografía del lector (si está disponible)
    if readers is not None:
        r = readers[["genero", "nacimiento"]].rename(columns={"genero": "reader_sex"})
        df = df.join(r, on=USER_COL)
        df["reader_sex"] = df["reader_sex"].map({"Hombre": 0, "Mujer": 1}).astype("float32")
        df["reader_age"] = ref.year - df["nacimiento"]
        df = df.drop(columns=["nacimiento"])

    return df.drop(columns=["autor", "editorial", "genero"])


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (USER_COL, ITEM_COL, "label")]
