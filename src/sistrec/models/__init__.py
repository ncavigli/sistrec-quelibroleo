from .base import Recommender, interaction_weights
from .content import AttributeAffinity, NewItemAffinity
from .ease import EASE
from .hybrid import WeightedHybrid
from .neighborhood import RP3beta, ItemKNN
from .popularity import Popularity
from .sequential import Sequential, SeriesContinuation


def __getattr__(name):  # ALS/BPR importan `implicit` sólo si se usan
    if name in ("ALS", "BPR"):
        from . import factorization

        return getattr(factorization, name)
    raise AttributeError(name)


__all__ = [
    "Recommender", "interaction_weights", "Popularity", "ItemKNN", "RP3beta", "EASE",
    "AttributeAffinity", "NewItemAffinity", "WeightedHybrid", "Sequential", "SeriesContinuation", "ALS", "BPR",
]
