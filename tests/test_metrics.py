import math

import pytest

from sistrec.metrics import evaluate, ndcg_at_k, recall_at_k


def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=20) == pytest.approx(1.0)


def test_ndcg_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=20) == 0.0


def test_ndcg_single_hit_position_2():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a"}, k=20) == pytest.approx(1 / math.log2(3))


def test_ndcg_respects_cutoff():
    assert ndcg_at_k(["x", "a"], {"a"}, k=1) == 0.0


def test_ndcg_ideal_capped_at_k():
    # 3 relevantes pero K=2: el ideal sólo cuenta 2 posiciones
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2) == pytest.approx(1.0)


def test_ndcg_ignores_duplicates():
    assert ndcg_at_k(["a", "a", "b"], {"b"}, k=2) == pytest.approx(1 / math.log2(3))


def test_graded():
    s = ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}, k=2)
    dcg = 1 + 2 / math.log2(3)
    idcg = 2 + 1 / math.log2(3)
    assert s == pytest.approx(dcg / idcg)


def test_recall_and_evaluate():
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5
    res = evaluate({"u1": ["a"], "u2": []}, {"u1": {"a"}, "u2": {"b"}}, k=20)
    assert res["ndcg@20"] == pytest.approx(0.5)
    assert res["n_users"] == 2
