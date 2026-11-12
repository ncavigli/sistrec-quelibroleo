# sistrec-quelibroleo

Sistema de recomendación de libros para [quelibroleo.com](https://quelibroleo.com/), desarrollado para la
competencia Kaggle **FCEN DM – Sistemas de Recomendación 2026** (Maestría en Data Mining).

**Tarea:** para cada uno de los 832 lectores de `ejemplo.csv`, recomendar 20 libros que *va a leer*.
**Métrica:** NDCG@20.

## Hallazgos clave del análisis de datos

| | |
|---|---|
| Interacciones | 461.408 (lector, libro, fecha, rating 1–10), 2008 → **31-12-2024** |
| Lectores / libros con interacciones | 10.673 / 48.137 (catálogo: 128.743) |
| Lectores a predecir | 832 × 20 libros; 16 sin historial (cold start) |
| Historial de los lectores objetivo | mediana 92 libros (vs. 9 del total) |

1. **El test es el futuro.** Los datos se cortan el 31-12-2024 y los lectores objetivo generan el 80% de las
   lecturas de 2024. Por eso la validación es **temporal** (train < 2024, verdad = lecturas de 2024), no aleatoria.
2. **~22% de lo que se lee en un año son novedades** que no existían en el historial previo: un modelo
   colaborativo puro no puede recomendarlas.
3. **El catálogo tiene ~715 libros editados en 2025–2026 sin interacciones.** Los sumamos como candidatos
   (`--new-items`) y los recomendamos por afinidad de autor/género.
4. Ya leído ≠ recomendable: se excluyen siempre los libros que el lector ya tiene en su historial.

## Arquitectura

```
                 ┌─ popularidad 90d / 365d ───┐
                 ├─ afinidad autor / género ───┤
historial ──────►├─ ItemKNN / EASE / ALS ──────┤──► ~380 candidatos ──► features ──► LightGBM ──► top-20
(< corte)        ├─ secuencial (orden lectura) ┤    por lector          (~50)       LambdaRank
catálogo ───────►├─ continuación de sagas ─────┤
                 └─ novedades del catálogo ────┘
```

## Estructura

```
src/sistrec/
  config.py        rutas y constantes
  data.py          carga desde SQLite, InteractionMatrix, novedades del catálogo
  split.py         split temporal para validación
  metrics.py       NDCG@K, recall@K, hit rate
  evaluation.py    corre y registra experimentos
  submission.py    escribe y valida el CSV
  registry.py      configuraciones de modelos por nombre
  series.py        detección de saga + tomo en los títulos
  models/
    base.py          interfaz común (exclusión de vistos, cold start, top-K)
    popularity.py    popularidad (con decaimiento temporal opcional)
    neighborhood.py  ItemKNN (coseno) y RP3beta (random walk)
    ease.py          EASE^R (autoencoder lineal, solución cerrada)
    factorization.py ALS / BPR (librería implicit)
    content.py       afinidad por autor / género / novedades
    sequential.py    transiciones entre lecturas y continuación de sagas
    hybrid.py        ensamble ponderado
  ranking/
    candidates.py    etapa 1: candidatos de varios modelos + sus puntajes/posiciones
    features.py      etapa 2: features de libro, autor, lector y lector x libro
    pipeline.py      folds temporales + LightGBM LambdaRank
scripts/
  validate.py         compara modelos en el split temporal
  make_submission.py  submission con un modelo individual
  train_ranker.py     pipeline completo de 2 etapas: valida en 2024 y genera la submission
  tune_ranker.py      compara versiones de candidatos / configs del ranker con varias semillas
tests/                tests de métricas y de modelos
```

## Uso

```bash
# 1. Entorno (Python 3.12)
py -3.12 -m venv ~/.venvs/recsys-libros
~/.venvs/recsys-libros/Scripts/python -m pip install -e ".[dev]"

# 2. Datos: copiar data.db y ejemplo.csv a data/raw/ (no se versionan)

# 3. Validar modelos
python scripts/validate.py --cutoff 2024-01-01 --models pop rp3beta itemknn autor --new-items

# 4. Ranker de 2 etapas (mejor modelo): valida y genera submissions/<fecha>_ranker.csv
python scripts/train_ranker.py            # ~15 min la primera vez; luego usa folds cacheados
python scripts/train_ranker.py --rebuild  # si cambiaste candidatos o features

# 4b. Submission con un modelo individual
python scripts/make_submission.py --model autor --new-items

# 5. Tests
python -m pytest
```

## Resultados de validación

| Modelo | NDCG@20 (validación 2024) |
|---|---|
| **Ranker 2 etapas v3** (+ secuencial, sagas, 5 años) — **Kaggle público 0.12650** | **0.0907** |
| Ranker 2 etapas v1 (Kaggle público: 0.11658) | 0.0827 |
| Afinidad autor + novedades | 0.0535 |
| ItemKNN con decaimiento | 0.0525 |
| Popularidad reciente | 0.0387 |
| Popularidad histórica | 0.0203 |


Ver [`docs/RESULTADOS.md`](docs/RESULTADOS.md).
