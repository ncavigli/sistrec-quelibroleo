"""Etapa 2: features para cada par (lector, libro) candidato.

Todas se calculan SÓLO con el `train` del fold (fecha < cutoff) para no filtrar el futuro.

Grupos:
  - Libro:          popularidad en ventanas recientes, tendencia, rating medio, antigüedad, novedad.
  - Autor:          popularidad reciente del autor.
  - Lector:         actividad, recencia, rating medio, gusto por novedades.
  - Lector x libro: cuántos libros leyó del autor/género/editorial, cuándo fue la última vez,
                    qué rating le dio al autor, distancia al año de edición que suele leer.
  - Sagas:          tomo del libro, cuántos tomos de la saga leyó, si es el siguiente.
  - Alta:           días desde que el libro se publicó en el sitio; si el lector lee novedades.
  - Ediciones:      si ya leyó otra edición del mismo libro.
  - Texto:          similitud entre el resumen del libro y lo que el lector viene leyendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

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


def _text_similarity(df: pd.DataFrame, train: pd.DataFrame, emb: pd.DataFrame) -> pd.DataFrame:
    """Coseno entre el embedding del libro y el perfil de texto del lector.

    text_sim_long: perfil = últimas 50 lecturas (peso 0.95^k); text_sim_short: últimas 5.
    """
    M = emb.to_numpy(np.float32)
    pos = pd.Series(np.arange(len(emb)), index=emb.index)
    t = train.sort_values([USER_COL, "fecha"], kind="stable")
    t = t[t[ITEM_COL].isin(pos.index)]
    users = pd.Index(df[USER_COL].unique())
    t = t[t[USER_COL].isin(users)]
    out = {}
    for name, n, decay in (("text_sim_long", 50, 0.95), ("text_sim_short", 5, 0.8)):
        last = t.groupby(USER_COL, sort=False).tail(n)
        k = last.groupby(USER_COL, sort=False).cumcount(ascending=False).to_numpy()
        R = sp.csr_matrix(
            (np.power(decay, k).astype(np.float32),
             (users.get_indexer(last[USER_COL]), pos.reindex(last[ITEM_COL]).to_numpy())),
            shape=(len(users), len(emb)),
        )
        P = np.asarray(R @ M)
        P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        ui = users.get_indexer(df[USER_COL])
        ii = pos.reindex(df[ITEM_COL]).to_numpy()
        ok = ~np.isnan(ii)
        sim = np.full(len(df), np.nan, dtype=np.float32)
        sim[ok] = np.einsum("ij,ij->i", P[ui[ok]], M[ii[ok].astype(np.int64)])
        has_profile = np.asarray(R.sum(axis=1)).ravel() > 0
        sim[~has_profile[ui]] = np.nan
        out[name] = sim
    return pd.DataFrame(out, index=df.index)


def build_features(
    cands: pd.DataFrame,
    train: pd.DataFrame,
    books: pd.DataFrame,
    readers: pd.DataFrame | None = None,
    text_emb: pd.DataFrame | None = None,
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

    # --- fecha de alta en el sitio (timestamp de la tapa): los libros se leen apenas se publican
    if "fecha_alta" in books:
        alta = books["fecha_alta"].reindex(df[ITEM_COL]).to_numpy()
        df["item_days_since_alta"] = ((ref - pd.DatetimeIndex(alta)).days).astype("float32")
        # el lector, ¿suele leer libros recién dados de alta?
        ra = books["fecha_alta"].reindex(train[ITEM_COL]).to_numpy()
        lag = (pd.DatetimeIndex(train["fecha"]) - pd.DatetimeIndex(ra)).days.to_numpy(float)
        fresh = np.where(np.isnan(lag), np.nan, (lag <= 90).astype(float))  # leído <= 90 días del alta
        by_user = train[USER_COL].to_numpy()
        df["user_frac_read_fresh"] = df[USER_COL].map(pd.Series(fresh).groupby(by_user).mean())
        df["user_alta_lag_median"] = df[USER_COL].map(pd.Series(lag).groupby(by_user).median())

    # --- otra edición de un libro que ya leyó (mismo título + autor, otro id)
    if "obra" in books:
        obra = books["obra"]
        uo = set(zip(train[USER_COL], obra.reindex(train[ITEM_COL]).to_numpy()))
        cand_obra = obra.reindex(df[ITEM_COL]).to_numpy()
        df["ua_otra_edicion"] = np.fromiter(
            ((u, o) in uo for u, o in zip(df[USER_COL], cand_obra)), dtype=np.int8, count=len(df)
        )
        df.loc[pd.isna(cand_obra), "ua_otra_edicion"] = 0

    # --- similitud de texto entre el libro y lo que el lector viene leyendo
    if text_emb is not None:
        df = df.join(_text_similarity(df, train, text_emb))

    # --- demografía del lector (si está disponible)
    if readers is not None:
        r = readers[["genero", "nacimiento"]].rename(columns={"genero": "reader_sex"})
        df = df.join(r, on=USER_COL)
        df["reader_sex"] = df["reader_sex"].map({"Hombre": 0, "Mujer": 1}).astype("float32")
        df["reader_age"] = ref.year - df["nacimiento"]
        df = df.drop(columns=["nacimiento"])

    df = df.drop(columns=["autor", "editorial", "genero"])
    # float32 para que entren más folds en memoria (la precisión sobra para árboles)
    f64 = df.select_dtypes("float64").columns
    df[f64] = df[f64].astype("float32")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (USER_COL, ITEM_COL, "label")]
