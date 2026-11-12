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
