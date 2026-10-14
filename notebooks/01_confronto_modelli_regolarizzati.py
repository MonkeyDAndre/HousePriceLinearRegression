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
