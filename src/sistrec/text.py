"""Embeddings de texto de los libros (título + resumen).

TF-IDF sobre el texto -> TruncatedSVD a `dim` dimensiones (LSA) -> normalización L2.
Así dos libros con resúmenes parecidos quedan cerca aunque nadie los haya leído juntos, lo que
sirve sobre todo para las novedades. Sólo usa el texto del catálogo (no hay etiquetas
involucradas), así que no filtra información del futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import DATA_PROCESSED, SEED

# Palabras vacías frecuentes en castellano (sklearn sólo trae inglés).
_STOP = """a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estas este
esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mis mucho muy nada ni no
nos nuestra nuestro o os otra otras otro otros para pero poco por porque que quien se sea ser
si sin sobre su sus tambien te tiene tienen todo todos tu un una uno unos y ya yo libro novela
historia vida autor obra""".split()


def book_embeddings(books: pd.DataFrame, dim: int = 64, use_cache: bool = True) -> pd.DataFrame:
    """DataFrame (id_libro x dim) float32 con filas de norma 1; sólo libros con texto útil."""
    # la clave depende del catálogo: un catálogo distinto (p.ej. en tests) no reutiliza el caché
    key = pd.util.hash_pandas_object(books.index.to_series(), index=False).sum() % 10**10
    path = DATA_PROCESSED / f"text_emb_{dim}_{key}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    text = (books["titulo"].fillna("") + ". " + books["resumen"].fillna("")).str.strip()
    text = text[books["resumen"].fillna("").str.len() > 30]
    vec = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, stop_words=_STOP, min_df=3, max_df=0.3,
        max_features=60000, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )
    X = vec.fit_transform(text.to_numpy())
    Z = TruncatedSVD(n_components=dim, random_state=SEED).fit_transform(X)
    Z = normalize(Z).astype(np.float32)
    emb = pd.DataFrame(Z, index=text.index, columns=[f"t{i}" for i in range(dim)])
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        emb.to_parquet(path)
    return emb
