"""sistrec: sistema de recomendación de libros para quelibroleo.com."""

import os

# OpenBLAS precompilado soporta hasta 24 hilos; con más (ej. i9 de 32 hilos) puede crashear
# en silencio. Debe fijarse ANTES de importar numpy, por eso los scripts importan sistrec primero.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

__version__ = "0.1.0"
