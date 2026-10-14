import numpy as np
import pandas as pd
import pytest


def test_costanti(ns):
    assert ns["RANDOM_STATE"] == 42
    assert ns["DATA_URL"] == "https://proai-datasets.s3.eu-west-3.amazonaws.com/housing.csv"
    assert ns["FEATURE_COLS"] == ns["NUMERIC_COLS"] + ns["BINARY_COLS"] + [ns["FURNISH_COL"]]
    assert len(ns["FEATURE_COLS"]) == 12
    assert ns["FURNISH_CORRETTO"] == {0: 0, 1: 2, 2: 1}
    assert ns["ENCODINGS"] == ["ordinal_ingenuo", "ordinal_corretto", "onehot"]
    assert ns["NOMI_MODELLI"] == ["Lineare", "Ridge", "Lasso", "ElasticNet"]
    assert ns["INR_PER_EUR"] == pytest.approx(107.6305)


def test_preprocessore_ordinal_ingenuo_lascia_i_codici(ns, df_sintetico):
    X = df_sintetico[ns["FEATURE_COLS"]]
    out = ns["costruisci_preprocessore"]("ordinal_ingenuo").fit_transform(X)
    assert list(out.columns) == ns["FEATURE_COLS"]
    assert (out["furnishingstatus"].values == X["furnishingstatus"].values).all()


def test_preprocessore_ordinal_corretto_ricodifica(ns, df_sintetico):
    X = df_sintetico[ns["FEATURE_COLS"]]
    out = ns["costruisci_preprocessore"]("ordinal_corretto").fit_transform(X)
    attesi = X["furnishingstatus"].map({0: 0, 1: 2, 2: 1}).values
    assert (out["furnishingstatus"].values == attesi).all()


def test_preprocessore_onehot_riferimento_non_arredato(ns, df_sintetico):
    X = df_sintetico[ns["FEATURE_COLS"]]
    out = ns["costruisci_preprocessore"]("onehot").fit_transform(X)
    assert out.shape[1] == 13
    assert {"furnishingstatus_1", "furnishingstatus_2"} <= set(out.columns)
    assert "furnishingstatus_0" not in out.columns
    righe_non_arredate = (X["furnishingstatus"] == 0).values
    assert (out.loc[righe_non_arredate, ["furnishingstatus_1", "furnishingstatus_2"]].values == 0).all()


def test_preprocessore_encoding_sconosciuto(ns):
    with pytest.raises(ValueError):
        ns["costruisci_preprocessore"]("target_encoding")


def test_pipeline_struttura_e_log_target(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    pipe = ns["costruisci_pipeline"]("onehot", "Ridge")
    assert list(pipe.named_steps) == ["prep", "scaler", "model"]
    ttr = ns["costruisci_pipeline"]("onehot", "Ridge", log_target=True).fit(X, y)
    assert type(ttr).__name__ == "TransformedTargetRegressor"
    assert (ttr.predict(X) > 0).all()
    assert list(ns["pipeline_interna"](ttr).named_steps) == ["prep", "scaler", "model"]


def test_crea_modello_restituisce_istanze_nuove(ns):
    assert ns["crea_modello"]("Lasso") is not ns["crea_modello"]("Lasso")
    with pytest.raises(ValueError):
        ns["crea_modello"]("SVR")
