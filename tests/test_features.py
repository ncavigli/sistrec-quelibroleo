import numpy as np
import pandas as pd

from sistrec.ranking.features import _text_similarity


def test_text_similarity_profiles():
    emb = pd.DataFrame([[1, 0], [0, 1], [0.6, 0.8]], index=["a", "b", "c"],
                       columns=["t0", "t1"], dtype=np.float32)
    train = pd.DataFrame({"id_lector": ["u", "u"], "id_libro": ["a", "a"],
                          "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    cands = pd.DataFrame({"id_lector": ["u", "u", "u", "v"], "id_libro": ["a", "b", "zzz", "a"]})
    sim = _text_similarity(cands, train, emb)
    assert sim.loc[0, "text_sim_long"] == np.float32(1.0)    # mismo libro que su perfil
    assert abs(sim.loc[1, "text_sim_long"]) < 1e-6          # ortogonal
    assert np.isnan(sim.loc[2, "text_sim_long"])             # libro sin texto
    assert np.isnan(sim.loc[3, "text_sim_short"])            # lector sin historial
