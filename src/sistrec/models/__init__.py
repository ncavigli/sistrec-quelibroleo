from .base import Recommender, interaction_weights
from .content import AttributeAffinity
from .ease import EASE
from .hybrid import WeightedHybrid
from .neighborhood import RP3beta, ItemKNN
from .popularity import Popularity


def __getattr__(name):  # ALS/BPR importan `implicit` sólo si se usan
    if name in ("ALS", "BPR"):
        from . import factorization

        return getattr(factorization, name)
    raise AttributeError(name)


__all__ = [
    "Recommender", "interaction_weights", "Popularity", "ItemKNN", "RP3beta", "EASE",
    "AttributeAffinity", "WeightedHybrid", "ALS", "BPR",
]
