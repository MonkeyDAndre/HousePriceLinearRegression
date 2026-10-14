# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stima dei prezzi residenziali: confronto tra modelli regolarizzati
#
# **RealEstateAI Solutions – Team Data Science**
#
# ## Il problema
# Il motore di valutazione attuale si basa su una regressione lineare multipla. In addestramento
# funziona bene, ma sugli immobili nuovi le stime si discostano più del previsto: il sospetto è
# l'**overfitting**, cioè un modello che si adatta troppo ai dati visti e generalizza male.
#
# ## La domanda
# La **regolarizzazione** (Ridge, Lasso, Elastic Net) migliora le stime? Quali caratteristiche
# dell'immobile pesano davvero sul prezzo? Il modello sbaglia sistematicamente su qualche fascia di prezzo?
#
# ## Come leggere questo notebook
# Ogni blocco di codice è preceduto da una spiegazione (*cosa facciamo e perché*) e seguito da un
# commento (*cosa osserviamo*). Chi non scrive codice può leggere solo le parti di testo.
#
# ## Indice
# 1. Librerie e impostazioni
# 2. Funzioni di supporto
# 3. Lettura e pulizia del dataset
# 4. Analisi esplorativa
# 5. Split train/test e modello di riferimento (baseline)
# 6. Modelli regolarizzati con tre codifiche delle variabili categoriche
# 7. Confronto e scelta della codifica
# 8. Passaggio al logaritmo del prezzo
# 9. Prezzo originale vs logaritmo
# 10. Valutazione finale sul test set
# 11. Conclusioni e raccomandazione

# %% [markdown]
# ## 1. Librerie e impostazioni
# Importiamo tutte le librerie usate nel notebook e fissiamo le costanti del progetto, in
# particolare il seme casuale `RANDOM_STATE` che rende i risultati riproducibili.

# %% tags=["imports"]
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold, cross_val_predict,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Riproducibilità
RANDOM_STATE = 42

# Dati
DATA_URL = "https://proai-datasets.s3.eu-west-3.amazonaws.com/housing.csv"
TARGET = "price"
NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
FURNISH_COL = "furnishingstatus"
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + [FURNISH_COL]

# furnishingstatus: significato dei codici nel CSV e ricodifica nell'ordine corretto
FURNISH_ETICHETTE = {0: "non arredato", 1: "arredato", 2: "semi-arredato"}
FURNISH_CORRETTO = {0: 0, 1: 2, 2: 1}  # codice CSV -> 0 non arredato, 1 semi, 2 arredato

# Unità di misura
SQFT_PER_M2 = 10.7639
INR_PER_EUR = 107.6305          # tasso di riferimento BCE
TASSO_FONTE = "BCE, 26/06/2026"

# Esperimenti
ENCODINGS = ["ordinal_ingenuo", "ordinal_corretto", "onehot"]
NOMI_MODELLI = ["Lineare", "Ridge", "Lasso", "ElasticNet"]
# Ordini di preferenza in caso di risultati equivalenti (vedi sezione 2b)
ORDINE_ENCODING = ["ordinal_corretto", "onehot", "ordinal_ingenuo"]
ORDINE_MODELLI = ["Lasso", "Ridge", "ElasticNet", "Lineare"]
ORDINE_TARGET = ["prezzo", "log(prezzo)"]

# Stile dei grafici
sns.set_theme(style="whitegrid", context="notebook")
COLORI_MODELLI = dict(zip(NOMI_MODELLI, sns.color_palette("colorblind", len(NOMI_MODELLI))))
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# %% [markdown]
# ## 2. Funzioni di supporto
# Raccogliamo qui tutte le funzioni usate nel resto del notebook. Usare **le stesse funzioni per
# ogni esperimento** garantisce che tutti i modelli siano preparati e valutati con lo stesso
# protocollo: le differenze nei risultati dipendono dai modelli, non da dettagli di codice.
#
# ### 2a. Preparazione dei dati
# - **Tre codifiche** di `furnishingstatus` (l'unica categorica con 3 livelli; le altre sono già 0/1):
#   - `ordinal_ingenuo`: i codici del CSV così come sono (0 non arredato, 1 arredato, 2 semi-arredato),
#     cioè con un **ordine sbagliato**;
#   - `ordinal_corretto`: ricodifica nell'ordine naturale 0 non arredato → 1 semi → 2 arredato;
#   - `onehot`: una colonna 0/1 per "arredato" e una per "semi-arredato", con "non arredato" come riferimento.
# - **Standardizzazione** di tutte le feature (media 0, deviazione standard 1): la penalità della
#   regolarizzazione dipende dalla scala delle variabili, quindi le mettiamo tutte sullo stesso metro.
# - Tutto dentro una **Pipeline**: in cross-validation la standardizzazione viene calcolata solo sui
#   dati di training di ciascun fold, evitando che informazioni della validazione "trapelino" nel modello.

# %% tags=["helpers"]
def crea_modello(nome):
    """Restituisce una nuova istanza (non addestrata) del modello richiesto."""
    if nome == "Lineare":
        return LinearRegression()
    if nome == "Ridge":
        return Ridge()
    if nome == "Lasso":
        return Lasso(max_iter=100_000)
    if nome == "ElasticNet":
        return ElasticNet(max_iter=100_000)
    raise ValueError(f"Modello sconosciuto: {nome}")


def _ricodifica_arredamento(X):
    """Riporta furnishingstatus nell'ordine naturale: 0 non arredato, 1 semi, 2 arredato."""
    X = X.copy()
    X[FURNISH_COL] = X[FURNISH_COL].map(FURNISH_CORRETTO)
    return X


def costruisci_preprocessore(encoding):
    """Trasforma le 12 feature secondo la codifica scelta per furnishingstatus."""
    if encoding == "ordinal_ingenuo":
        trasformazione_arredo = "passthrough"
    elif encoding == "ordinal_corretto":
        trasformazione_arredo = FunctionTransformer(_ricodifica_arredamento, feature_names_out="one-to-one")
    elif encoding == "onehot":
        trasformazione_arredo = OneHotEncoder(categories=[[0, 1, 2]], drop="first", sparse_output=False)
    else:
        raise ValueError(f"Codifica sconosciuta: {encoding}")
    preprocessore = ColumnTransformer(
        [("numeriche", "passthrough", NUMERIC_COLS),
         ("binarie", "passthrough", BINARY_COLS),
         ("arredamento", trasformazione_arredo, [FURNISH_COL])],
        verbose_feature_names_out=False,
    )
    return preprocessore.set_output(transform="pandas")


def costruisci_pipeline(encoding, nome_modello, log_target=False):
    """Codifica -> standardizzazione -> modello; con log_target il modello impara log(prezzo)
    e le predizioni vengono riportate automaticamente in rupie."""
    pipeline = Pipeline([
        ("prep", costruisci_preprocessore(encoding)),
        ("scaler", StandardScaler()),
        ("model", crea_modello(nome_modello)),
    ])
    if log_target:
        return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp,
                                          check_inverse=False)
    return pipeline


def pipeline_interna(estimatore):
    """Restituisce la Pipeline addestrata, anche se avvolta in un TransformedTargetRegressor."""
    return estimatore.regressor_ if isinstance(estimatore, TransformedTargetRegressor) else estimatore

# %% [markdown]
# ### 2b. Addestramento e valutazione
# - **Split stratificato**: mettiamo da parte il 20% dei dati come *test set*, mantenendo in entrambe
#   le parti la stessa proporzione di case economiche, medie e costose (5 fasce di prezzo).
# - **Cross-validation ripetuta** (5 fold × 3 ripetizioni = 15 valutazioni): ogni configurazione viene
#   addestrata e valutata 15 volte su porzioni diverse del training set; riportiamo media e deviazione
#   standard dell'errore. Il CSV è ordinato per prezzo, quindi i dati vengono **mescolati** prima di dividerli.
# - **Ricerca degli iperparametri**: per ogni modello proviamo molti valori della forza di regolarizzazione
#   `alpha` (e, per Elastic Net, del mix `l1_ratio` tra Lasso e Ridge) e teniamo il migliore.
# - **Metriche**: MSE (errore quadratico medio), RMSE (radice dell'MSE, in rupie: "errore tipico"),
#   MAE (errore assoluto medio, in rupie), R² (quota di variabilità del prezzo spiegata dal modello).
# - **Regola di scelta**: vince l'RMSE medio più basso; se un'alternativa dista meno di una deviazione
#   standard, i due risultati sono **equivalenti** e preferiamo la soluzione più semplice/interpretabile
#   secondo un ordine dichiarato in anticipo:
#   - codifiche: ordinal corretto → one-hot → ordinal ingenuo (meno colonne prima; quella con ordine
#     sbagliato per ultima);
#   - modelli: Lasso (meno coefficienti, più leggibile) → Ridge (un solo iperparametro) → Elastic Net
#     (due iperparametri) → Lineare (nessuna protezione dall'overfitting, che è il problema di partenza);
#   - target: prezzo → log(prezzo) (nessuna trasformazione da spiegare).
# - **Effetti reali**: i coefficienti di un modello addestrato su feature standardizzate indicano l'effetto
#   di "+1 deviazione standard". Dividendo per la deviazione standard otteniamo l'effetto per unità reale
#   (una camera in più, avere l'aria condizionata, un piede quadrato in più) in rupie e in euro.

# %% tags=["helpers"]
GRIGLIE_ALPHA = {
    ("Ridge", False): np.logspace(-3, 4, 30),
    ("Ridge", True): np.logspace(-3, 4, 30),
    ("Lasso", False): np.logspace(0, 6, 30),
    ("Lasso", True): np.logspace(-5, 0, 30),
    ("ElasticNet", False): np.logspace(0, 6, 20),
    ("ElasticNet", True): np.logspace(-5, 0, 20),
}
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
SCORING = {"mse": "neg_mean_squared_error", "rmse": "neg_root_mean_squared_error",
           "mae": "neg_mean_absolute_error", "r2": "r2"}
UNITA_FEATURE = {"area": "per sq ft", "bedrooms": "per camera in più", "bathrooms": "per bagno in più",
                 "stories": "per piano in più", "parking": "per posto auto in più",
                 "furnishingstatus": "per livello di arredamento"}
NOMI_LEGGIBILI = {"furnishingstatus_1": "arredamento: arredato (vs non arredato)",
                  "furnishingstatus_2": "arredamento: semi-arredato (vs non arredato)"}


def split_stratificato(df, test_size=0.2, n_fasce=5):
    """Split train/test mescolato e stratificato per fasce (quantili) di prezzo."""
    fasce = pd.qcut(df[TARGET], q=n_fasce, labels=False)
    return train_test_split(df[FEATURE_COLS], df[TARGET], test_size=test_size, shuffle=True,
                            stratify=fasce, random_state=RANDOM_STATE)


def crea_cv():
    """5-fold cross-validation ripetuta 3 volte, con mescolamento."""
    return RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)


def _prefisso(log_target):
    return "regressor__model__" if log_target else "model__"


def griglia_parametri(nome_modello, log_target=False):
    """Iperparametri da esplorare per ciascun modello."""
    p = _prefisso(log_target)
    if nome_modello == "Lineare":
        return {p + "fit_intercept": [True]}  # nessun iperparametro: una sola configurazione
    griglia = {p + "alpha": GRIGLIE_ALPHA[(nome_modello, log_target)]}
    if nome_modello == "ElasticNet":
        griglia[p + "l1_ratio"] = L1_RATIOS
    return griglia


def metriche(y_true, y_pred):
    return {"mse": mean_squared_error(y_true, y_pred), "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


def conta_coef_nonzero(estimatore, tol=1e-8):
    coef = pipeline_interna(estimatore).named_steps["model"].coef_
    return int(np.sum(np.abs(coef) > tol))


def alpha_al_bordo(search):
    """True se l'alpha migliore è il primo o l'ultimo valore della griglia (griglia da allargare)."""
    chiave = next((k for k in search.best_params_ if k.endswith("alpha")), None)
    if chiave is None:
        return False
    griglia = search.param_grid[chiave]
    migliore = search.best_params_[chiave]
    return bool(np.isclose(migliore, griglia[0]) or np.isclose(migliore, griglia[-1]))


def addestra_configurazione(X, y, encoding, nome_modello, log_target=False):
    """Ricerca degli iperparametri in cross-validation; restituisce metriche e modello migliore."""
    search = GridSearchCV(costruisci_pipeline(encoding, nome_modello, log_target),
                          griglia_parametri(nome_modello, log_target),
                          scoring=SCORING, refit="mse", cv=crea_cv(), n_jobs=-1)
    search.fit(X, y)
    i, cv = search.best_index_, search.cv_results_
    bordo = alpha_al_bordo(search)
    if bordo:
        warnings.warn(f"{nome_modello} / {encoding}: alpha ottimo al bordo della griglia, allargarla.")
    return {
        "encoding": encoding,
        "modello": nome_modello,
        "target": "log(prezzo)" if log_target else "prezzo",
        "rmse_mean": -cv["mean_test_rmse"][i],
        "rmse_std": cv["std_test_rmse"][i],
        "mse_mean": -cv["mean_test_mse"][i],
        "mae_mean": -cv["mean_test_mae"][i],
        "r2_mean": cv["mean_test_r2"][i],
        "n_coef_nonzero": conta_coef_nonzero(search.best_estimator_),
        "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()
                        if not k.endswith("fit_intercept")},
        "alpha_al_bordo": bordo,
        "search": search,
    }


def tabella_risultati(risultati):
    colonne = ["encoding", "modello", "target", "rmse_mean", "rmse_std", "mse_mean", "mae_mean",
               "r2_mean", "n_coef_nonzero", "best_params", "alpha_al_bordo"]
    righe = [{c: r[c] for c in colonne} for r in risultati]
    return pd.DataFrame(righe).sort_values("rmse_mean").reset_index(drop=True)


def trova_risultato(risultati, encoding, modello, target):
    return next(r for r in risultati
                if (r["encoding"], r["modello"], r["target"]) == (encoding, modello, target))


def scegli_migliore(df, colonna, ordine_preferenza):
    """Regola di scelta: tra le righe entro 1 deviazione standard dal migliore,
    vince quella che compare prima in ordine_preferenza."""
    migliore = df.loc[df["rmse_mean"].idxmin()]
    soglia = migliore["rmse_mean"] + migliore["rmse_std"]
    equivalenti = df[df["rmse_mean"] <= soglia].copy()
    equivalenti["_rango"] = equivalenti[colonna].map(ordine_preferenza.index)
    return equivalenti.sort_values(["_rango", "rmse_mean"]).iloc[0].drop("_rango")


def predizioni_oof(estimatore, X, y):
    """Predizioni 'fuori campione' sul training set: ogni riga è predetta da un modello che non l'ha vista."""
    return cross_val_predict(estimatore, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                             n_jobs=-1)


def errore_per_fascia(y_true, y_pred, n_fasce=5):
    """Residuo medio (reale - predetto) e MAE per fascia di prezzo reale."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fasce = pd.qcut(y_true, q=n_fasce)
    df = pd.DataFrame({"fascia": fasce, "residuo": y_true - y_pred, "errore_ass": np.abs(y_true - y_pred)})
    tab = df.groupby("fascia", observed=True).agg(n=("residuo", "size"), residuo_medio=("residuo", "mean"),
                                                  mae=("errore_ass", "mean")).reset_index()
    tab["fascia"] = tab["fascia"].astype(str)
    return tab


def _unita(nome_feature):
    if nome_feature in BINARY_COLS:
        return "0 → 1"
    if nome_feature in NOMI_LEGGIBILI:
        return "vs non arredato"
    return UNITA_FEATURE.get(nome_feature, "")


def tabella_effetti(estimatore, log_target=False, prezzo_riferimento=None):
    """Coefficienti standardizzati ed effetto reale di ogni feature.
    - prezzo: effetto in rupie e in euro per unità reale;
    - log(prezzo): effetto percentuale; se prezzo_riferimento è dato, anche l'equivalente in euro
      su una casa di quel prezzo."""
    pipeline = pipeline_interna(estimatore)
    nomi = list(pipeline.named_steps["prep"].get_feature_names_out())
    coef = pipeline.named_steps["model"].coef_
    scala = pipeline.named_steps["scaler"].scale_
    effetto = coef / scala

    tab = pd.DataFrame({"feature": nomi, "unita": [_unita(n) for n in nomi],
                        "coef_std": coef, "effetto_unita": effetto})
    i_area = nomi.index("area")
    riga_m2 = {"feature": "area (per m²)", "unita": "per m²", "coef_std": np.nan,
               "effetto_unita": effetto[i_area] * SQFT_PER_M2}
    tab = pd.concat([tab, pd.DataFrame([riga_m2])], ignore_index=True)

    if log_target:
        tab["effetto_pct"] = (np.exp(tab["effetto_unita"]) - 1) * 100
        if prezzo_riferimento is not None:
            tab["effetto_eur"] = tab["effetto_pct"] / 100 * prezzo_riferimento / INR_PER_EUR
    else:
        tab["effetto_inr"] = tab["effetto_unita"]
        tab["effetto_eur"] = tab["effetto_inr"] / INR_PER_EUR
    tab["feature"] = tab["feature"].map(lambda n: NOMI_LEGGIBILI.get(n, n))
    tab = tab.drop(columns="effetto_unita")
    ordine = tab["coef_std"].abs().fillna(-1).sort_values(ascending=False).index
    return tab.loc[ordine].reset_index(drop=True)
