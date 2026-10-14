from types import SimpleNamespace

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


def test_split_stratificato(ns, df_sintetico):
    X_tr, X_te, y_tr, y_te = ns["split_stratificato"](df_sintetico)
    assert len(X_te) == 30 and len(X_tr) == 120
    assert list(X_tr.columns) == ns["FEATURE_COLS"]
    fasce = pd.qcut(df_sintetico["price"], 5, labels=False)
    assert fasce.loc[y_te.index].nunique() == 5


def test_crea_cv_15_valutazioni(ns, df_sintetico):
    assert ns["crea_cv"]().get_n_splits(df_sintetico) == 15


def test_griglia_parametri(ns):
    g = ns["griglia_parametri"]("ElasticNet", log_target=True)
    assert set(g) == {"regressor__model__alpha", "regressor__model__l1_ratio"}
    assert set(ns["griglia_parametri"]("Lineare")) == {"model__fit_intercept"}
    assert set(ns["griglia_parametri"]("Ridge")) == {"model__alpha"}


def test_metriche_predizione_perfetta(ns):
    y = np.array([1.0, 2.0, 3.0])
    m = ns["metriche"](y, y)
    assert m == {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 1.0}


def test_alpha_al_bordo(ns):
    griglia = {"model__alpha": [1.0, 10.0, 100.0]}
    assert ns["alpha_al_bordo"](SimpleNamespace(best_params_={"model__alpha": 1.0}, param_grid=griglia))
    assert not ns["alpha_al_bordo"](SimpleNamespace(best_params_={"model__alpha": 10.0}, param_grid=griglia))
    assert not ns["alpha_al_bordo"](SimpleNamespace(best_params_={"model__fit_intercept": True},
                                                     param_grid={"model__fit_intercept": [True]}))


def test_conta_coef_nonzero_lasso_alpha_enorme(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    pipe = ns["costruisci_pipeline"]("onehot", "Lasso").set_params(model__alpha=1e9).fit(X, y)
    assert ns["conta_coef_nonzero"](pipe) == 0


@pytest.mark.parametrize("nome", ["Lineare", "Ridge", "Lasso", "ElasticNet"])
def test_addestra_configurazione(ns, df_sintetico, nome):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    r = ns["addestra_configurazione"](X, y, "ordinal_corretto", nome)
    assert r["modello"] == nome and r["target"] == "prezzo"
    assert r["rmse_mean"] > 0 and r["rmse_std"] >= 0
    assert r["mse_mean"] >= 0.99 * r["rmse_mean"] ** 2  # media dei quadrati ≥ quadrato della media
    assert r["r2_mean"] > 0.5
    assert 0 <= r["n_coef_nonzero"] <= 12


def test_addestra_configurazione_log(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    r = ns["addestra_configurazione"](X, y, "onehot", "Ridge", log_target=True)
    assert r["target"] == "log(prezzo)"
    assert r["rmse_mean"] > 1e4  # metriche in rupie, non in scala log


def test_scegli_migliore_usa_preferenza_tra_equivalenti(ns):
    df = pd.DataFrame({"modello": ["Ridge", "Lasso", "Lineare"],
                       "rmse_mean": [100.0, 105.0, 200.0], "rmse_std": [10.0, 5.0, 1.0]})
    assert ns["scegli_migliore"](df, "modello", ["Lasso", "Ridge", "Lineare"])["modello"] == "Lasso"
    df.loc[1, "rmse_mean"] = 115.0  # fuori dalla soglia 100 + 10
    assert ns["scegli_migliore"](df, "modello", ["Lasso", "Ridge", "Lineare"])["modello"] == "Ridge"


def test_predizioni_oof_e_errore_per_fascia(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    pred = ns["predizioni_oof"](ns["costruisci_pipeline"]("onehot", "Ridge"), X, y)
    assert pred.shape == y.shape
    tab = ns["errore_per_fascia"](y, pred)
    assert list(tab.columns) == ["fascia", "n", "residuo_medio", "mae"]
    assert len(tab) == 5 and tab["n"].sum() == len(y)


def test_tabella_effetti_prezzo(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    pipe = ns["costruisci_pipeline"]("ordinal_corretto", "Lineare").fit(X, y)
    tab = ns["tabella_effetti"](pipe).set_index("feature")
    assert tab.loc["airconditioning", "effetto_inr"] == pytest.approx(4e5, abs=2e5)
    assert tab.loc["area", "effetto_inr"] == pytest.approx(300, abs=100)
    assert tab.loc["area (per m²)", "effetto_inr"] == pytest.approx(tab.loc["area", "effetto_inr"] * 10.7639)
    assert tab.loc["airconditioning", "effetto_eur"] == pytest.approx(tab.loc["airconditioning", "effetto_inr"] / 107.6305)
    assert tab.loc["airconditioning", "unita"] == "0 → 1"


def test_tabella_effetti_log(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    ttr = ns["costruisci_pipeline"]("onehot", "Ridge", log_target=True).fit(X, y)
    tab = ns["tabella_effetti"](ttr, log_target=True, prezzo_riferimento=4e6).set_index("feature")
    assert "effetto_pct" in tab.columns and "effetto_eur" in tab.columns
    riga = tab.loc["arredamento: arredato (vs non arredato)"]
    assert riga["effetto_eur"] == pytest.approx(riga["effetto_pct"] / 100 * 4e6 / 107.6305)
