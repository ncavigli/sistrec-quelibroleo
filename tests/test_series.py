import pandas as pd

from sistrec.series import parse_series


def test_parse_series_formats():
    t = pd.Series([
        "ANSIA (Serie Crave 3)",
        "OFRENDA A LA TORMENTA. Trilogía del Baztán 3",
        "EL LIBRO DE NOAH: SAGA VANIR VIII",
        "STAR WARS OMNIBUS. RELATOS JEDI Nº 01",
        "CUENTOS COMPLETOS (1864-1910)",
        "LA CANCIÓN NÚMERO SIETE (7)",
        "CIEN AÑOS DE SOLEDAD",
    ])
    s = parse_series(t)
    assert s.loc[0, "serie"] == "crave" and s.loc[0, "tomo"] == 3
    assert s.loc[1, "serie"] == "baztan" and s.loc[1, "tomo"] == 3
    assert s.loc[2, "serie"] == "vanir" and s.loc[2, "tomo"] == 8
    assert s.loc[3, "serie"] == "relatos jedi" and s.loc[3, "tomo"] == 1
    assert s.loc[4:6, "serie"].isna().all()
