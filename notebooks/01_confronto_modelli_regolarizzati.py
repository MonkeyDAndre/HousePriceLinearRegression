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
# Ordini di preferenza in caso di risultati equivalenti (vedi sezione 5)
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
