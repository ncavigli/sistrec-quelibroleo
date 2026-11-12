# Resultados de validación

**Protocolo:** split temporal con corte 2024-01-01. Train = todo lo anterior (446.906 interacciones).
Verdad = libros *nuevos para el lector* que leyó en 2024 (1.154 lectores; 644 de ellos están en el
test de Kaggle → columna `target`). Métrica principal: NDCG@20 con relevancia binaria.

## v0.1 – Baselines (2026-11-12)

| Modelo | NDCG@20 | NDCG@20 target | Recall@20 | Hit@20 |
|---|---|---|---|---|
| Afinidad autor + novedades (`autor --new-items`) | **0.0535** | **0.0668** | 0.0624 | 0.312 |
| Afinidad autor | 0.0517 | 0.0667 | 0.0604 | 0.308 |
| Afinidad género | 0.0438 | 0.0540 | 0.0605 | 0.274 |
| Popularidad reciente (vida media 180 d) | 0.0387 | 0.0464 | 0.0577 | 0.263 |
| ItemKNN coseno | 0.0346 | 0.0387 | 0.0433 | 0.228 |
| Popularidad histórica | 0.0203 | 0.0237 | 0.0286 | 0.161 |
| RP3beta (β=0.5) | 0.0104 | 0.0097 | 0.0147 | 0.081 |

**Lectura:**
- La popularidad *reciente* duplica a la histórica → fuerte deriva temporal (modas, novedades).
- El contenido (autor, género) le gana al colaborativo puro: el lector sigue a sus autores.
- RP3beta con β=0.5 castiga demasiado la popularidad, que acá es una señal valiosa.
- Conclusión: la mejor estrategia es un **ranker de 2 etapas** (generar candidatos con varios
  modelos + LightGBM LambdaRank que aprende a combinarlos con features de recencia, autor, género, etc.).

Submission `v01_autor_newitems.csv` generada (sin subir aún).

## v0.2 – Modelos individuales con decaimiento temporal

| Modelo | NDCG@20 | NDCG@20 target |
|---|---|---|
| ItemKNN con decaimiento (vida media 2 años) | 0.0525 | 0.0668 |
| EASE^R (8.000 libros más leídos, λ=500) | 0.0391 | 0.0481 |
| ALS (128 factores) | 0.0328 | 0.0391 |
| RP3beta con decaimiento | 0.0117 | 0.0127 |

El decaimiento temporal sube ItemKNN de 0.0346 a 0.0525 (+52%): lo que leíste hace poco dice
mucho más de lo que vas a leer que lo que leíste hace 10 años.

## v1 – Ranker de 2 etapas (LightGBM LambdaRank)

**Etapa 1 – candidatos** (~350/lector): pop90 (100), pop365 (50), autor (150), autor sin
popularidad (50), género (100), ItemKNN (100), EASE (100), ALS (100) + novedades del catálogo.

**Etapa 2 – ranker:** LightGBM `lambdarank` (optimiza NDCG@20) con ~45 features.

**Folds temporales:** train = folds 2022 y 2023 (features con datos < año, etiquetas = lecturas
nuevas de ese año); validación = fold 2024; final = re-entrenado con 2022+2023+2024, aplicado
a features con todo el historial.

| | NDCG@20 | NDCG@20 target | Recall@20 | Hit@20 |
|---|---|---|---|---|
| **Ranker v1** | **0.0827** | **0.1077** | 0.0978 | 0.427 |
| Mejor modelo individual (autor) | 0.0535 | 0.0668 | 0.0624 | 0.312 |

**+55% sobre el mejor modelo individual.** Recall de candidatos ≈ 0.27–0.30: es el techo
actual (el 70% de lo que se lee no está entre los candidatos), así que la próxima mejora
grande pasa por generar mejores candidatos.

Features más importantes (gain): antigüedad del libro en el sitio (`item_days_since_first`),
cuántos generadores lo propusieron (`n_generators`), puntaje EASE, posición en el ranking por
autor, edad del lector, rating relativo que le da al autor, popularidad total del autor.

> Nota: el número de árboles se eligió con early stopping sobre el mismo fold 2024 que se
> reporta, así que 0.0827 es levemente optimista.

Submission: `v02_ranker_v1.csv` (2.366 libros distintos, 336 recomendaciones son novedades
sin interacciones previas).

## Leaderboard público de Kaggle

| Submission | Validación (target) | Kaggle público |
|---|---|---|
| `v02_ranker_v1.csv` | 0.1077 | **0.11658** |

La validación con los lectores del test (0.108) predice bien el score público (0.117): el
protocolo temporal está bien calibrado y los supuestos (relevancia binaria, test = 2025, excluir
ya leídos) son correctos.

## v2 – Candidatos ampliados + generador de novedades

Diagnóstico de los relevantes de 2024 que *no* estaban entre los candidatos v1:

| Tipo de libro relevante | % | Recall candidatos v1 |
|---|---|---|
| Existía en el historial | 78% | 33.5% |
| Novedad del catálogo (ed. ≥ año-1, sin lecturas) | 14% | 8.2% |
| Novedad más vieja (no incluida) | 8% | 0% |

Cambios: generador `NewItemAffinity` (sólo puntúa libros sin lecturas, por afinidad de
autor/género/editorial + popularidad reciente del autor), ventana de novedades a 3 años, más
cupo para ItemKNN (200), EASE (150) y autor puro (100).

Recall de candidatos 2024: 0.273 → **0.339**, pero NDCG@20 no mejora (0.0827 → 0.0814).

### Comparación controlada (3 semillas, `scripts/tune_ranker.py`)

| Candidatos | base | small_leaves | lr05_l15 | deep_reg |
|---|---|---|---|---|
| v1 | 0.0833 | 0.0833 | 0.0839 | **0.0840** |
| v2 | 0.0831 | 0.0828 | 0.0830 | 0.0836 |

Desvío entre semillas ≈ 0.0005–0.001; error estándar por muestreo de lectores ≈ 0.004.
**Meseta en ~0.083–0.084**: ni más candidatos ni tuning mueven la métrica. Los relevantes que
agregó v2 son "difíciles" (entran como candidatos pero el ranker no tiene información para
subirlos). Las próximas mejoras tienen que aportar señal nueva (secuencias/sagas, más folds).

Submission `v03_ranker_v2.csv` generada; no vale la pena gastar un envío: es equivalente a v02.

Kaggle confirmó la validación: `v03_ranker_v2.csv` dio **peor** que `v02` en el leaderboard
público (validación target: 0.1036 vs 0.1077). Se descarta v2 y se vuelve a la base v1.

## v3 – Señal secuencial + sagas + más años de entrenamiento

**Generadores nuevos:**
- `Sequential`: transiciones dirigidas "después de leer i se lee j" (próximas 10 lecturas, ≤ 365
  días; mismo día = ambas direcciones). Perfil = últimas 30 lecturas con decaimiento 0.9^k.
  Solo: NDCG 0.0471 (2 s de entrenamiento, vs 80 s de ItemKNN con 0.0525).
- `SeriesContinuation`: detecta saga + tomo en el título (`series.py`; 1.878 libros, 801 sagas,
  9.2% de las lecturas) y recomienda el tomo siguiente. Solo: 0.0168 (aplica a pocos lectores).

**Features nuevas:** tomo del libro, tomos leídos de esa saga, tomo máximo leído, días desde la
última lectura de la saga, distancia al tomo siguiente.

**Folds de entrenamiento:** 2019–2023 (antes 2022–2023).

### Ablación (2 semillas, `tune_ranker.py`)

| Candidatos | Folds train | base | deep_reg |
|---|---|---|---|
| v1 | 2022–23 | 0.0833 | 0.0840 |
| v3 | 2022–23 | 0.0871 | 0.0880 |
| v3 | 2019–23 | 0.0874 | **0.0911** |

- Secuencial + sagas: **+0.004**. Más años: **+0.003**, pero sólo con un ranker de mayor
  capacidad y más regularizado (`deep_reg`: 127 hojas, min_child 300, λ=10, colsample 0.5),
  que pasa a ser el default.

### Resultado final v3

| | NDCG@20 | NDCG@20 target | Recall@20 | Hit@20 |
|---|---|---|---|---|
| **Ranker v3 deep_reg** | **0.0907** | **0.1179** | 0.1076 | 0.453 |
| Ranker v1 | 0.0827 | 0.1077 | 0.0978 | 0.427 |

370 recomendaciones son el tomo siguiente de una saga en curso. 68% de coincidencia con v02.

## Historial de submissions

| Archivo | Qué es | Validación target | Kaggle público |
|---|---|---|---|
| `ejemplo.csv` | ejemplo de la cátedra (mismos 20 libros para todos) | – | 0.00301 |
| `v01_autor_newitems.csv` | afinidad por autor | 0.0668 | – |
| `v02_ranker_v1.csv` | ranker v1 | 0.1077 | 0.11658 |
| `v03_ranker_v2.csv` | ranker + candidatos v2 | 0.1036 | 0.11288 |
| `v04_ranker_v3.csv` | ranker v3, params base | 0.1170 | 0.12562 |
| `v05_ranker_v3_deepreg.csv` | **ranker v3 deep_reg (final)** | **0.1179** | **0.12650** |

La validación local ordenó las 4 submissions del ranker exactamente igual que el leaderboard
público. Kaggle da ~0.009 más que la validación de forma consistente.

Diapositiva de presentación: `docs/presentacion/` (fuente) y el artifact publicado.
