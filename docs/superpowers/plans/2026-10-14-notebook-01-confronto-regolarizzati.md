# Notebook 01 – Confronto modelli regolarizzati: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produrre il notebook di consegna `notebooks/01_confronto_modelli_regolarizzati.ipynb`, eseguito e funzionante su Colab, che confronta regressione lineare, Ridge, Lasso ed Elastic Net con tre encoding e con target prezzo / log-prezzo.

**Architecture:** La sorgente del notebook è un file Jupytext in formato *percent* (`notebooks/01_confronto_modelli_regolarizzati.py`), comodo da editare e da diffare; l'`.ipynb` si genera con `jupytext` e si esegue con `nbconvert`. Le celle di import e di funzioni di supporto sono marcate con i tag `imports` / `helpers`: i test pytest le estraggono ed eseguono in un namespace, così le funzioni sono testate senza uscire dal notebook (che resta autosufficiente per Colab). Le sezioni narrative usano solo le funzioni di supporto.

**Tech Stack:** Python 3.13 (locale) / Colab, pandas, numpy, scikit-learn ≥1.5, scipy, matplotlib, seaborn, jupytext, pytest, nbconvert.

**Spec:** `docs/superpowers/specs/2026-10-14-regressione-regolarizzata-design.md`

**Fuori da questo piano:** il notebook personale `02_analisi_estesa.ipynb` avrà un piano separato, scritto dopo il completamento di questo (riusa le funzioni di supporto finali e confronta con il modello vincente del notebook 01).

## Global Constraints

- Lingua di notebook, markdown, nomi delle funzioni e commenti: **italiano**.
- `RANDOM_STATE = 42` ovunque serva casualità.
- Dataset letto **solo** da `https://proai-datasets.s3.eu-west-3.amazonaws.com/housing.csv`; nessun path locale nel notebook.
- Nessun modulo `.py` importato dal notebook; nessuna cella `%pip install` (tutte le librerie sono preinstallate su Colab).
- Unità originali: prezzo in **rupie indiane (INR)**, area in **piedi quadrati (sq ft)**; `1 m² = 10.7639 sq ft`.
- Conversione in euro solo negli effetti: `INR_PER_EUR = 107.6305` (tasso di riferimento BCE del 26/06/2026), dichiarato nel notebook con la nota "la conversione cambia l'unità, non il mercato".
- Codifica CSV di `furnishingstatus`: **0 = non arredato, 1 = arredato, 2 = semi-arredato**. Ordine corretto: 0 non arredato, 1 semi, 2 arredato.
- Ogni split/CV con `shuffle=True` (il CSV è ordinato per prezzo decrescente).
- Split: 80/20 stratificato su 5 quantili del prezzo. CV: `RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)`.
- Pipeline: `encoding → StandardScaler (tutte le feature, binarie incluse) → modello`.
- Il **test set si usa solo nella sezione 10**. La diagnosi di overfitting della baseline (sezione 5) usa un set di validazione ricavato dal train (vedi Task 6).
- Regola di scelta: minimo `rmse_mean` di CV; righe con `rmse_mean ≤ rmse_mean_min + rmse_std_min` sono equivalenti → si sceglie per ordine di preferenza dichiarato.
- Ogni grafico: titolo, assi con unità, seguito da una cella markdown "**Cosa osserviamo**" con 1–3 frasi basate sui numeri effettivi.
- Comandi eseguiti da Git Bash dalla root del repo; interprete: `.venv/Scripts/python.exe`.

## File Structure

| File | Responsabilità |
|---|---|
| `requirements.txt` | dipendenze dell'ambiente locale |
| `notebooks/01_confronto_modelli_regolarizzati.py` | **sorgente** del notebook (Jupytext percent) |
| `notebooks/01_confronto_modelli_regolarizzati.ipynb` | notebook generato ed eseguito, da caricare su Colab |
| `tests/nb_loader.py` | estrae ed esegue le celle taggate di un notebook Jupytext |
| `tests/conftest.py` | backend matplotlib non interattivo, fixture namespace e dataset sintetico |
| `tests/test_nb01_helpers.py` | test delle funzioni di supporto |

Convenzioni del formato percent (valide per tutto il piano):

```python
# %% [markdown]
# ## Titolo della cella markdown
# Testo su più righe, ogni riga preceduta da "# ".

# %% tags=["helpers"]
def funzione():
    ...
```

---

### Task 1: Ambiente, scheletro del notebook, loader dei test

**Files:**
- Create: `requirements.txt`, `notebooks/01_confronto_modelli_regolarizzati.py`, `tests/nb_loader.py`, `tests/conftest.py`, `tests/test_nb01_helpers.py`

**Interfaces:**
- Produces: `carica_namespace(percorso: Path, tag: tuple[str, ...] = ("imports", "helpers")) -> dict`; fixture pytest `ns` (dict namespace del notebook, scope session) e `df_sintetico` (DataFrame con le 13 colonne del dataset, 150 righe); costanti del notebook elencate nello Step 5.

- [ ] **Step 1: Creare `requirements.txt`**

```text
numpy>=2.0
pandas>=2.2
scipy>=1.13
scikit-learn>=1.5
matplotlib>=3.8
seaborn>=0.13
statsmodels>=0.14
jupyter>=1.0
ipykernel>=6.29
nbconvert>=7.16
jupytext>=1.16
pytest>=8.0
```

- [ ] **Step 2: Creare il venv e installare**

Run:
```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install --upgrade pip && .venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -c "import sklearn, pandas, jupytext; print(sklearn.__version__, pandas.__version__, jupytext.__version__)"
```
Expected: stampa tre versioni, scikit-learn ≥ 1.5.

- [ ] **Step 3: Creare `tests/nb_loader.py`**

```python
"""Estrae ed esegue le celle di codice taggate di un notebook Jupytext."""
from pathlib import Path

import jupytext


def carica_namespace(percorso: Path, tag: tuple[str, ...] = ("imports", "helpers")) -> dict:
    notebook = jupytext.read(percorso)
    # __name__ non importabile: cloudpickle (usato da n_jobs=-1) serializza le funzioni per valore
    namespace: dict = {"__name__": "notebook_sotto_test"}
    for cella in notebook.cells:
        tag_cella = set(cella.metadata.get("tags", []))
        if cella.cell_type == "code" and tag_cella & set(tag):
            exec(compile(cella.source, str(percorso), "exec"), namespace)
    return namespace
```

- [ ] **Step 4: Creare `tests/conftest.py`**

```python
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest

from nb_loader import carica_namespace

RADICE = Path(__file__).resolve().parents[1]
NB01 = RADICE / "notebooks" / "01_confronto_modelli_regolarizzati.py"


@pytest.fixture(scope="session")
def ns():
    return carica_namespace(NB01)


@pytest.fixture(scope="session")
def df_sintetico():
    """Dataset finto con la stessa struttura di housing.csv e effetti noti."""
    rng = np.random.default_rng(0)
    n = 150
    df = pd.DataFrame({
        "area": rng.integers(1650, 16200, n),
        "bedrooms": rng.integers(1, 7, n),
        "bathrooms": rng.integers(1, 5, n),
        "stories": rng.integers(1, 5, n),
        "mainroad": rng.binomial(1, 0.86, n),
        "guestroom": rng.binomial(1, 0.18, n),
        "basement": rng.binomial(1, 0.35, n),
        "hotwaterheating": rng.binomial(1, 0.05, n),
        "airconditioning": rng.binomial(1, 0.32, n),
        "parking": rng.integers(0, 4, n),
        "prefarea": rng.binomial(1, 0.23, n),
        "furnishingstatus": rng.choice([0, 1, 2], n),
    })
    effetto_arredo = df["furnishingstatus"].map({0: 0.0, 1: 4e5, 2: 2e5})
    prezzo = (1e6 + 300 * df["area"] + 2e5 * df["bathrooms"] + 4e5 * df["airconditioning"]
              + 3e5 * df["prefarea"] + effetto_arredo + rng.normal(0, 3e5, n))
    df.insert(0, "price", prezzo.clip(lower=1e6))
    return df
```

Non serve `tests/__init__.py`: `sys.path.insert` rende importabile `nb_loader`.

- [ ] **Step 5: Scrivere il test delle costanti (fallirà: il notebook non esiste)**

`tests/test_nb01_helpers.py`:

```python
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
```

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: ERROR (file del notebook non trovato).

- [ ] **Step 6: Creare il notebook con intestazione, introduzione e cella librerie**

`notebooks/01_confronto_modelli_regolarizzati.py`:

```python
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
```

- [ ] **Step 7: Eseguire i test**

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: `test_costanti PASSED`.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt notebooks tests
git commit -m "Ambiente, scheletro del notebook 01 e loader dei test"
```

---

### Task 2: Funzioni di supporto – preprocessing (sezione 2a)

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py` (aggiungere in coda)
- Test: `tests/test_nb01_helpers.py`

**Interfaces:**
- Consumes: costanti del Task 1.
- Produces:
  - `crea_modello(nome: str) -> estimator` (nuova istanza non addestrata; `nome` in `NOMI_MODELLI`)
  - `costruisci_preprocessore(encoding: str) -> ColumnTransformer` (output pandas; ordine colonne: numeriche, binarie, arredamento)
  - `costruisci_pipeline(encoding: str, nome_modello: str, log_target: bool = False) -> Pipeline | TransformedTargetRegressor` con step `"prep"`, `"scaler"`, `"model"`
  - `pipeline_interna(estimatore) -> Pipeline` (la Pipeline addestrata, anche se avvolta in TransformedTargetRegressor)

- [ ] **Step 1: Scrivere i test**

Aggiungere a `tests/test_nb01_helpers.py`:

```python
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
```

- [ ] **Step 2: Verificare che falliscano**

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: i 6 nuovi test FAIL con `KeyError: 'costruisci_preprocessore'` (o simile).

- [ ] **Step 3: Implementare (aggiungere in coda al notebook)**

```python
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
```

- [ ] **Step 4: Eseguire i test**

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: tutti PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks tests
git commit -m "Funzioni di supporto: preprocessing e pipeline"
```

---

### Task 3: Funzioni di supporto – addestramento e valutazione (sezione 2b)

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`
- Test: `tests/test_nb01_helpers.py`

**Interfaces:**
- Consumes: `costruisci_pipeline`, `pipeline_interna`, costanti.
- Produces:
  - `split_stratificato(df, test_size=0.2, n_fasce=5) -> (X_train, X_test, y_train, y_test)`
  - `crea_cv() -> RepeatedKFold`
  - `GRIGLIE_ALPHA: dict[(str, bool), np.ndarray]`, `L1_RATIOS: list[float]`
  - `griglia_parametri(nome_modello, log_target=False) -> dict`
  - `metriche(y_true, y_pred) -> dict` con chiavi `mse, rmse, mae, r2`
  - `conta_coef_nonzero(estimatore, tol=1e-8) -> int`
  - `alpha_al_bordo(search) -> bool`
  - `addestra_configurazione(X, y, encoding, nome_modello, log_target=False) -> dict` con chiavi `encoding, modello, target ("prezzo" | "log(prezzo)"), rmse_mean, rmse_std, mse_mean, mae_mean, r2_mean, n_coef_nonzero, best_params, alpha_al_bordo, search`
  - `tabella_risultati(risultati: list[dict]) -> pd.DataFrame` (senza `search`, ordinata per `rmse_mean`)
  - `trova_risultato(risultati, encoding, modello, target) -> dict`
  - `scegli_migliore(df, colonna, ordine_preferenza) -> pd.Series`
  - `predizioni_oof(estimatore, X, y) -> np.ndarray` (predizioni out-of-fold, KFold 5)
  - `errore_per_fascia(y_true, y_pred, n_fasce=5) -> pd.DataFrame` con colonne `fascia, n, residuo_medio, mae`
  - `tabella_effetti(estimatore, log_target=False, prezzo_riferimento=None) -> pd.DataFrame`

- [ ] **Step 1: Scrivere i test**

```python
from types import SimpleNamespace


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
```

- [ ] **Step 2: Verificare che falliscano**

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: nuovi test FAIL con `KeyError`.

- [ ] **Step 3: Implementare (aggiungere in coda al notebook)**

```python
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
```

Nota: nel test `test_tabella_effetti_prezzo` la riga `"area (per m²)"` ha `coef_std = NaN` ed è posta in fondo dall'ordinamento (`fillna(-1)`).

- [ ] **Step 4: Eseguire i test**

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: tutti PASS. Se compaiono `ConvergenceWarning` di Lasso/ElasticNet, verificare che riguardino solo gli alpha più piccoli; se un test fallisce per convergenza, portare `max_iter` a 500_000 in `crea_modello`.

- [ ] **Step 5: Commit**

```bash
git add notebooks tests
git commit -m "Funzioni di supporto: addestramento, valutazione ed effetti reali"
```

---

### Task 4: Funzioni di supporto – grafici (sezione 2c)

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`
- Test: `tests/test_nb01_helpers.py`

**Interfaces:**
- Consumes: `costruisci_preprocessore`, `crea_cv`, `GRIGLIE_ALPHA`, `errore_per_fascia`, `COLORI_MODELLI`, `NOMI_MODELLI`.
- Produces (tutte restituiscono `matplotlib.figure.Figure`):
  - `grafico_barre_modelli(df, metrica, ylabel, titolo, x="encoding", colonna_errore=None)`
  - `grafico_errore_vs_alpha(risultati, titolo)`
  - `grafico_percorso_lasso(X, y, encoding, titolo, log_target=False)`
  - `grafico_residui_vs_predetti(coppie: dict[str, tuple], titolo)`
  - `grafico_residui(y_true, y_pred, titolo)`
  - `grafico_predetto_vs_reale(y_true, y_pred, titolo)`
  - `grafico_learning_curve(estimatore, X, y, titolo)`
  - `grafico_effetti(tabella, colonna, xlabel, titolo)`

- [ ] **Step 1: Scrivere i test (smoke: ogni funzione produce una Figure)**

```python
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@pytest.fixture(scope="module")
def risultati_sintetici(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    return [ns["addestra_configurazione"](X, y, enc, nome)
            for enc in ["ordinal_corretto", "onehot"] for nome in ns["NOMI_MODELLI"]]


def _verifica(fig):
    assert isinstance(fig, Figure)
    assert len(fig.axes) > 0
    plt.close(fig)


def test_grafico_barre_modelli(ns, risultati_sintetici):
    df = ns["tabella_risultati"](risultati_sintetici)
    _verifica(ns["grafico_barre_modelli"](df, "rmse_mean", "RMSE (INR)", "t", colonna_errore="rmse_std"))
    _verifica(ns["grafico_barre_modelli"](df, "n_coef_nonzero", "n", "t"))


def test_grafico_errore_vs_alpha(ns, risultati_sintetici):
    _verifica(ns["grafico_errore_vs_alpha"](risultati_sintetici[:4], "t"))


def test_grafico_percorso_lasso(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    _verifica(ns["grafico_percorso_lasso"](X, y, "onehot", "t"))
    _verifica(ns["grafico_percorso_lasso"](X, y, "onehot", "t", log_target=True))


def test_grafici_residui(ns, df_sintetico):
    y = df_sintetico["price"].values
    pred = y + np.random.default_rng(1).normal(0, 1e5, len(y))
    _verifica(ns["grafico_residui_vs_predetti"]({"a": (y, pred), "b": (y, pred)}, "t"))
    _verifica(ns["grafico_residui"](y, pred, "t"))
    _verifica(ns["grafico_predetto_vs_reale"](y, pred, "t"))


def test_grafico_learning_curve(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    _verifica(ns["grafico_learning_curve"](ns["costruisci_pipeline"]("onehot", "Ridge"), X, y, "t"))


def test_grafico_effetti(ns, df_sintetico):
    X, y = df_sintetico[ns["FEATURE_COLS"]], df_sintetico["price"]
    tab = ns["tabella_effetti"](ns["costruisci_pipeline"]("onehot", "Ridge").fit(X, y))
    _verifica(ns["grafico_effetti"](tab, "effetto_eur", "Effetto (€)", "t"))
```

- [ ] **Step 2: Verificare che falliscano**

Run: `.venv/Scripts/python.exe -m pytest tests -v -k grafic`
Expected: FAIL con `KeyError`.

- [ ] **Step 3: Implementare (aggiungere in coda al notebook)**

```python
# %% [markdown]
# ### 2c. Grafici
# Funzioni per i grafici che ripetiamo più volte (confronti tra modelli, residui, coefficienti), così
# tutti hanno lo stesso stile: stessi colori per lo stesso modello, titoli che dicono cosa si mostra,
# assi con le unità di misura.

# %% tags=["helpers"]
def grafico_barre_modelli(df, metrica, ylabel, titolo, x="encoding", colonna_errore=None):
    """Barre raggruppate: un gruppo per valore di `x`, una barra per modello."""
    gruppi = list(dict.fromkeys(df[x]))
    modelli = [m for m in NOMI_MODELLI if m in set(df["modello"])]
    larghezza = 0.8 / len(modelli)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, modello in enumerate(modelli):
        sotto = df[df["modello"] == modello].set_index(x).reindex(gruppi)
        posizioni = np.arange(len(gruppi)) + (j - (len(modelli) - 1) / 2) * larghezza
        errore = sotto[colonna_errore] if colonna_errore else None
        ax.bar(posizioni, sotto[metrica], larghezza, yerr=errore, capsize=3, label=modello,
               color=COLORI_MODELLI[modello])
    ax.set_xticks(np.arange(len(gruppi)), gruppi)
    ax.set(xlabel=x, ylabel=ylabel, title=titolo)
    ax.legend(title="Modello", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def grafico_errore_vs_alpha(risultati, titolo):
    """RMSE di cross-validation al variare di alpha, una curva per modello regolarizzato."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in risultati:
        if r["modello"] == "Lineare":
            continue
        cv = pd.DataFrame(r["search"].cv_results_)
        col_alpha = next(c for c in cv.columns if c.startswith("param_") and c.endswith("alpha"))
        col_l1 = next((c for c in cv.columns if c.startswith("param_") and c.endswith("l1_ratio")), None)
        if col_l1 is not None:  # ElasticNet: curva al miglior l1_ratio
            cv = cv[np.isclose(cv[col_l1].astype(float), r["best_params"]["l1_ratio"])]
        cv = cv.sort_values(col_alpha)
        alpha = cv[col_alpha].astype(float).values
        media, std = -cv["mean_test_rmse"].values, cv["std_test_rmse"].values
        colore = COLORI_MODELLI[r["modello"]]
        ax.plot(alpha, media, color=colore, label=r["modello"])
        ax.fill_between(alpha, media - std, media + std, color=colore, alpha=0.15)
        ax.scatter([r["best_params"]["alpha"]], [r["rmse_mean"]], color=colore, edgecolor="black", zorder=3, s=60)
    ax.set_xscale("log")
    ax.set(xlabel="alpha (forza della regolarizzazione, scala log)", ylabel="RMSE medio di CV (INR)", title=titolo)
    ax.legend(title="Modello (● = alpha scelto)")
    fig.tight_layout()
    return fig


def grafico_percorso_lasso(X, y, encoding, titolo, log_target=False):
    """Coefficienti del Lasso al crescere di alpha: le feature vengono azzerate una dopo l'altra."""
    preparazione = Pipeline([("prep", costruisci_preprocessore(encoding)), ("scaler", StandardScaler())])
    Xs = preparazione.fit_transform(X)
    y_modello = np.log(np.asarray(y)) if log_target else np.asarray(y, dtype=float)
    alphas, coefs, _ = lasso_path(Xs.values, y_modello - y_modello.mean(),
                                  alphas=GRIGLIE_ALPHA[("Lasso", log_target)], max_iter=100_000)
    fig, ax = plt.subplots(figsize=(10, 6))
    colori = sns.color_palette("tab20", len(Xs.columns))
    for i, nome in enumerate(Xs.columns):
        ax.plot(alphas, coefs[i], label=NOMI_LEGGIBILI.get(nome, nome), color=colori[i])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    unita = "log(prezzo)" if log_target else "INR"
    ax.set(xlabel="alpha (scala log)", ylabel=f"Coefficiente standardizzato ({unita})", title=titolo)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize="small")
    fig.tight_layout()
    return fig


def grafico_residui_vs_predetti(coppie, titolo):
    """Residui (reale - predetto) contro valore predetto, un pannello per ogni approccio."""
    fig, axes = plt.subplots(1, len(coppie), figsize=(6 * len(coppie), 5), sharey=True, squeeze=False)
    for ax, (nome, (y_true, y_pred)) in zip(axes[0], coppie.items()):
        residui = np.asarray(y_true) - np.asarray(y_pred)
        ax.scatter(y_pred, residui, alpha=0.5, s=15)
        ax.axhline(0, color="red", linestyle="--")
        ax.set(title=nome, xlabel="Prezzo predetto (INR)", ylabel="Residuo (INR)")
    fig.suptitle(titolo)
    fig.tight_layout()
    return fig


def grafico_residui(y_true, y_pred, titolo):
    """Pannello diagnostico: residui vs predetti, distribuzione dei residui, residuo medio per fascia di prezzo."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    residui = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_pred, residui, alpha=0.5, s=15)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set(title="Residui vs predetti", xlabel="Prezzo predetto (INR)", ylabel="Residuo (INR)")
    sns.histplot(residui, stat="density", ax=axes[1])
    griglia = np.linspace(residui.min(), residui.max(), 200)
    axes[1].plot(griglia, stats.norm.pdf(griglia, residui.mean(), residui.std()), color="red", label="normale")
    axes[1].set(title="Distribuzione dei residui", xlabel="Residuo (INR)")
    axes[1].legend()
    tab = errore_per_fascia(y_true, y_pred)
    axes[2].bar(tab["fascia"], tab["residuo_medio"], color=["#c44e52" if v < 0 else "#4c72b0" for v in tab["residuo_medio"]])
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(title="Residuo medio per fascia di prezzo reale", xlabel="Fascia di prezzo (INR)",
                ylabel="Residuo medio (INR)\n> 0 sottostima, < 0 sovrastima")
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle(titolo)
    fig.tight_layout()
    return fig


def grafico_predetto_vs_reale(y_true, y_pred, titolo):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=15)
    limiti = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(limiti, limiti, color="red", linestyle="--", label="predizione perfetta")
    ax.set(xlabel="Prezzo reale (INR)", ylabel="Prezzo predetto (INR)", title=titolo)
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_learning_curve(estimatore, X, y, titolo):
    """Errore di training e di validazione al crescere della quantità di dati di training."""
    dimensioni, train, valid = learning_curve(
        estimatore, X, y, cv=crea_cv(), scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 6), shuffle=True, random_state=RANDOM_STATE, n_jobs=-1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for punteggi, etichetta, colore in [(train, "training", "#4c72b0"), (valid, "validazione", "#dd8452")]:
        media, std = -punteggi.mean(axis=1), punteggi.std(axis=1)
        ax.plot(dimensioni, media, marker="o", label=etichetta, color=colore)
        ax.fill_between(dimensioni, media - std, media + std, alpha=0.15, color=colore)
    ax.set(xlabel="Numero di case nel training", ylabel="RMSE (INR)", title=titolo)
    ax.legend()
    fig.tight_layout()
    return fig


def grafico_effetti(tabella, colonna, xlabel, titolo):
    """Barre orizzontali degli effetti reali, ordinati per importanza (|coefficiente standardizzato|)."""
    dati = tabella.dropna(subset=["coef_std"]).iloc[::-1]
    etichette = [f"{f} ({u})" if u else f for f, u in zip(dati["feature"], dati["unita"])]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(dati) + 1.5))
    ax.barh(etichette, dati[colonna], color=["#c44e52" if v < 0 else "#4c72b0" for v in dati[colonna]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel=xlabel, title=titolo)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Eseguire i test**

Run: `.venv/Scripts/python.exe -m pytest tests -v`
Expected: tutti PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks tests
git commit -m "Funzioni di supporto: grafici"
```

---

### Task 5: Sezioni 3–4 – lettura, pulizia, analisi esplorativa + pipeline di build

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`
- Create: `notebooks/01_confronto_modelli_regolarizzati.ipynb` (generato)

**Interfaces:**
- Consumes: costanti, nessuna funzione.
- Produces: variabile notebook `df` (DataFrame del dataset, codici originali).

- [ ] **Step 1: Aggiungere la sezione 3**

```python
# %% [markdown]
# ## 3. Lettura e pulizia del dataset
# Carichiamo il dataset di compravendite direttamente dall'URL fornito dal team e controlliamo
# dimensioni, tipi delle colonne, valori mancanti e duplicati.

# %%
df = pd.read_csv(DATA_URL)
print(f"Righe: {df.shape[0]}, colonne: {df.shape[1]}")
df.head()

# %%
df.info()

# %%
print("Valori mancanti per colonna:")
print(df.isna().sum())
print(f"\nRighe duplicate: {df.duplicated().sum()}")

# %%
df.describe().T

# %% [markdown]
# **Cosa osserviamo** – <da scrivere dopo l'esecuzione, vedi Step 3>
#
# ### Unità di misura
# Il dataset proviene dal mercato indiano (dataset "Housing Prices" di Kaggle):
# - **prezzo** in **rupie indiane (INR)**;
# - **area** in **piedi quadrati (sq ft)**; 1 m² = 10,7639 sq ft.
#
# Addestriamo i modelli in unità originali: con le feature standardizzate l'unità di misura delle
# variabili non influenza il modello. Solo nella lettura finale degli effetti convertiremo le rupie
# in euro al tasso di riferimento BCE del 26/06/2026 (1 € = 107,6305 INR). Attenzione: la conversione
# cambia l'unità, non il mercato – restano prezzi del mercato indiano.
#
# ### Un controllo sulla codifica di `furnishingstatus`
# La documentazione ricevuta indica 0 = non arredato, 1 = parzialmente arredato, 2 = completamente
# arredato. Confrontiamo i conteggi con quelli del dataset Kaggle originale, dove le etichette sono testuali.

# %%
conteggi_csv = df[FURNISH_COL].value_counts().sort_index()
conteggi_kaggle = {"unfurnished (non arredato)": 178, "furnished (arredato)": 140,
                   "semi-furnished (semi-arredato)": 227}
pd.DataFrame({
    "codice nel CSV": conteggi_csv.index,
    "n. case nel CSV": conteggi_csv.values,
    "etichetta Kaggle con lo stesso conteggio": [k for v in conteggi_csv.values
                                                 for k, c in conteggi_kaggle.items() if c == v],
})

# %% [markdown]
# **Cosa osserviamo** – I conteggi coincidono uno a uno: nel CSV **1 = arredato** e **2 = semi-arredato**,
# cioè l'opposto di quanto indicato nella documentazione. Usare i codici così come sono, come variabile
# ordinale, significherebbe dire al modello che "semi-arredato" sta *oltre* "arredato". Nella sezione 6
# confronteremo proprio questa codifica "ingenua" con quella corretta e con il one-hot encoding, per
# misurare quanto costa un'assunzione di ordine sbagliata.
```

- [ ] **Step 2: Aggiungere la sezione 4 (analisi esplorativa)**

```python
# %% [markdown]
# ## 4. Analisi esplorativa
# Prima di addestrare qualsiasi modello guardiamo come sono fatti i dati: distribuzioni, relazioni
# con il prezzo e relazioni tra le feature. Queste osservazioni ci aiuteranno a interpretare i coefficienti.
#
# ### 4.1 Il prezzo

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df[TARGET], kde=True, ax=axes[0])
axes[0].set(title="Distribuzione del prezzo", xlabel="Prezzo (INR)", ylabel="Numero di case")
sns.histplot(np.log(df[TARGET]), kde=True, ax=axes[1], color="#dd8452")
axes[1].set(title="Distribuzione del logaritmo del prezzo", xlabel="log(prezzo)", ylabel="Numero di case")
fig.tight_layout()
plt.show()
print(f"Asimmetria (skewness) prezzo: {df[TARGET].skew():.2f} | log(prezzo): {np.log(df[TARGET]).skew():.2f}")

# %% [markdown]
# **Cosa osserviamo** – <da scrivere dopo l'esecuzione>
#
# ### 4.2 Le feature numeriche e categoriche

# %%
fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(20, 4))
for ax, col in zip(axes, NUMERIC_COLS):
    sns.histplot(df[col], ax=ax, discrete=col != "area")
    ax.set(title=col, xlabel=f"{col} (sq ft)" if col == "area" else col, ylabel="Numero di case")
fig.suptitle("Distribuzione delle feature numeriche")
fig.tight_layout()
plt.show()

# %%
categoriche = BINARY_COLS + [FURNISH_COL]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes.flat, categoriche):
    etichette = df[col].map(FURNISH_ETICHETTE) if col == FURNISH_COL else df[col].map({0: "no", 1: "sì"})
    etichette.value_counts().plot.bar(ax=ax, color="#4c72b0")
    ax.set(title=col, xlabel="", ylabel="Numero di case")
    ax.tick_params(axis="x", rotation=0)
axes.flat[-1].axis("off")
fig.suptitle("Frequenza delle feature categoriche")
fig.tight_layout()
plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere dopo l'esecuzione: asimmetria dell'area, categorie rare
# (es. hotwaterheating), sbilanciamenti>
#
# ### 4.3 Matrice di correlazione
# Per `furnishingstatus` usiamo la codifica nell'ordine corretto, così la correlazione ha senso.

# %%
corr = df.assign(**{FURNISH_COL: df[FURNISH_COL].map(FURNISH_CORRETTO)}).corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlazione di Pearson tra prezzo e feature")
fig.tight_layout()
plt.show()
corr[TARGET].drop(TARGET).sort_values(ascending=False).to_frame("correlazione con il prezzo")

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: feature più correlate al prezzo; coppie di feature correlate
# tra loro (collinearità) che possono rendere instabili i coefficienti della regressione lineare>
#
# ### 4.4 Prezzo e area

# %%
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df.assign(aria_condizionata=df["airconditioning"].map({0: "no", 1: "sì"})),
                x="area", y=TARGET, hue="aria_condizionata", alpha=0.7, ax=ax)
ax.set(title="Prezzo in funzione dell'area (colore: aria condizionata)", xlabel="Area (sq ft)", ylabel="Prezzo (INR)")
fig.tight_layout()
plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: relazione crescente, dispersione che aumenta con l'area/prezzo>
#
# ### 4.5 Prezzo per categoria

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.flat, categoriche):
    if col == FURNISH_COL:
        etichette = df[col].map(FURNISH_ETICHETTE)
        ordine = ["non arredato", "semi-arredato", "arredato"]
    else:
        etichette = df[col].map({0: "no", 1: "sì"})
        ordine = ["no", "sì"]
    sns.boxplot(x=etichette, y=df[TARGET], order=ordine, ax=ax)
    ax.set(title=col, xlabel="", ylabel="Prezzo (INR)")
axes.flat[-1].axis("off")
fig.suptitle("Distribuzione del prezzo per ciascuna feature categorica")
fig.tight_layout()
plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: quali categorie spostano la mediana del prezzo; per
# furnishingstatus, se la progressione non arredato → semi → arredato è monotona>
```

- [ ] **Step 3: Generare ed eseguire il notebook, poi scrivere le osservazioni**

Run:
```bash
.venv/Scripts/python.exe -m jupytext --to ipynb notebooks/01_confronto_modelli_regolarizzati.py
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_confronto_modelli_regolarizzati.ipynb
```
Expected: nessun errore. Aprire l'`.ipynb` (o leggere gli output testuali) e sostituire **ogni** segnaposto `<da scrivere ...>` nel file `.py` con 1–3 frasi basate sui numeri effettivi (es. "Il prezzo ha asimmetria X: poche case molto costose..."). Rigenerare ed eseguire di nuovo con gli stessi due comandi.

Verifica: `grep -n "da scrivere" notebooks/01_confronto_modelli_regolarizzati.py` → nessun risultato.

- [ ] **Step 4: Eseguire i test (le celle helper non devono essersi rotte)**

Run: `.venv/Scripts/python.exe -m pytest tests -q`
Expected: tutti PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks
git commit -m "Notebook 01: lettura, pulizia e analisi esplorativa"
```

---

### Task 6: Sezione 5 – split e baseline

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`

**Interfaces:**
- Consumes: `split_stratificato`, `costruisci_pipeline`, `metriche`.
- Produces: variabili `X_train, X_test, y_train, y_test`.

**Nota di design:** la consegna chiede di misurare la baseline "su training e test con hold-out". Per non consumare il test set (riservato alla sezione 10), il confronto hold-out della baseline usa un **set di validazione** (20% del training, stratificato con lo stesso criterio). Il markdown lo spiega.

- [ ] **Step 1: Aggiungere la sezione 5**

```python
# %% [markdown]
# ## 5. Split train/test e modello di riferimento (baseline)
# ### 5.1 Mettere da parte il test set
# Separiamo subito il **20% delle case come test set**: non lo useremo per nessuna scelta, ma solo alla
# fine (sezione 10) per misurare le prestazioni su case "mai viste". La divisione è stratificata per
# fasce di prezzo, così anche le case più costose sono rappresentate in entrambe le parti.

# %%
X_train, X_test, y_train, y_test = split_stratificato(df)
print(f"Training: {len(X_train)} case | Test: {len(X_test)} case")
pd.DataFrame({"training": y_train.describe(), "test": y_test.describe()})

# %% [markdown]
# ### 5.2 Baseline: regressione lineare multipla
# Addestriamo il modello attuale (regressione lineare senza regolarizzazione) e confrontiamo l'errore
# sui dati di addestramento con quello su dati non visti. Per non "sprecare" il test set usiamo un
# **set di validazione**: il 20% del training, messo da parte con lo stesso criterio stratificato.
# Se l'errore su dati nuovi è molto più alto di quello in addestramento, il modello è in overfitting.
# Usiamo la codifica corretta di `furnishingstatus`.

# %%
df_train = X_train.assign(**{TARGET: y_train})
X_fit, X_val, y_fit, y_val = split_stratificato(df_train)
baseline = costruisci_pipeline("ordinal_corretto", "Lineare").fit(X_fit, y_fit)
confronto_baseline = pd.DataFrame({
    "addestramento": metriche(y_fit, baseline.predict(X_fit)),
    "validazione": metriche(y_val, baseline.predict(X_val)),
})
confronto_baseline["rapporto validazione/addestramento"] = confronto_baseline["validazione"] / confronto_baseline["addestramento"]
confronto_baseline

# %%
fig, ax = plt.subplots(figsize=(8, 5))
confronto_baseline.loc[["rmse", "mae"], ["addestramento", "validazione"]].plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
ax.set(title="Baseline: errore in addestramento vs su dati non visti", ylabel="Errore (INR)", xlabel="")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: RMSE addestramento vs validazione e rapporto. Se il divario è
# ampio → overfitting confermato, la regolarizzazione è motivata. Se è piccolo → dirlo apertamente:
# con 545 case e 12 feature la regressione lineare ha poche occasioni di "memorizzare" i dati; la
# regolarizzazione resta utile per stabilizzare i coefficienti (feature correlate viste in 4.3) e per
# selezionare le feature, e lo verifichiamo nelle sezioni successive con la cross-validation>
#
# Un singolo split può essere fortunato o sfortunato: dalla prossima sezione valutiamo ogni modello con la
# **cross-validation ripetuta**, che usa tutte le case del training a turno come validazione.
```

- [ ] **Step 2: Rigenerare, eseguire, scrivere le osservazioni**

Run gli stessi due comandi del Task 5 Step 3; sostituire il segnaposto con il commento basato sui numeri; rigenerare ed eseguire di nuovo.
Verifica: `grep -n "da scrivere" notebooks/01_confronto_modelli_regolarizzati.py` → nessun risultato.

- [ ] **Step 3: Commit**

```bash
git add notebooks
git commit -m "Notebook 01: split stratificato e baseline"
```

---

### Task 7: Sezioni 6–7 – modelli × encoding, confronto e scelta

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`

**Interfaces:**
- Consumes: `addestra_configurazione`, `tabella_risultati`, `scegli_migliore`, `trova_risultato`, `tabella_effetti`, grafici del Task 4.
- Produces: variabili `risultati_prezzo: list[dict]`, `df_prezzo: DataFrame`, `ENCODING_SCELTO: str`.

- [ ] **Step 1: Aggiungere la sezione 6**

```python
# %% [markdown]
# ## 6. Modelli regolarizzati con tre codifiche
# La **regolarizzazione** aggiunge all'obiettivo del modello una penalità sulla grandezza dei coefficienti:
# - **Ridge** (penalità L2) "rimpicciolisce" tutti i coefficienti, senza azzerarli: utile con feature correlate;
# - **Lasso** (penalità L1) può portare **esattamente a zero** i coefficienti meno utili: seleziona le feature;
# - **Elastic Net** combina le due penalità (`l1_ratio` regola il mix).
#
# La forza della penalità è `alpha`: con alpha = 0 si torna alla regressione lineare, con alpha molto grande
# il modello predice sempre il prezzo medio. Per ciascuno dei 4 modelli e delle 3 codifiche cerchiamo
# l'alpha migliore con la cross-validation ripetuta: **12 configurazioni, stesso protocollo**.
#
# Ricordiamo che le 6 binarie sono identiche in tutte le codifiche: le differenze possono venire solo da
# `furnishingstatus`.

# %%
risultati_prezzo = [addestra_configurazione(X_train, y_train, encoding, nome)
                    for encoding in ENCODINGS for nome in NOMI_MODELLI]
df_prezzo = tabella_risultati(risultati_prezzo)
df_prezzo

# %% [markdown]
# **Una nota sull'MSE.** L'errore quadratico medio è nell'ordine di 10¹² perché si misura in
# *rupie al quadrato*: un numero corretto ma poco leggibile. Per questo affianchiamo l'RMSE (la sua radice,
# in rupie) e il MAE: dicono di quante rupie sbaglia, tipicamente, il modello su una casa.
#
# **Cosa osserviamo** – <da scrivere: RMSE migliore e peggiore, confronto con il prezzo medio
# (errore relativo ≈ RMSE / prezzo medio), eventuali alpha al bordo (colonna alpha_al_bordo)>
```

- [ ] **Step 2: Aggiungere la sezione 7**

```python
# %% [markdown]
# ## 7. Confronto e scelta della codifica
# ### 7.1 Errore di cross-validation

# %%
fig = grafico_barre_modelli(df_prezzo, "rmse_mean", "RMSE medio di CV (INR)",
                            "RMSE di cross-validation per codifica e modello (barre = ±1 dev. std)",
                            colonna_errore="rmse_std")
plt.show()
df_prezzo.pivot(index="encoding", columns="modello", values="rmse_mean")[NOMI_MODELLI].loc[ENCODINGS]

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: le differenze tra codifiche sono più grandi o più piccole delle
# barre di errore? Quanto peggiora la codifica ingenua?>
#
# ### 7.2 Scelta della codifica
# Per ogni codifica prendiamo il modello migliore e applichiamo la regola di scelta della sezione 2b.

# %%
migliori_per_encoding = df_prezzo.loc[df_prezzo.groupby("encoding")["rmse_mean"].idxmin()]
scelta_encoding = scegli_migliore(migliori_per_encoding, "encoding", ORDINE_ENCODING)
ENCODING_SCELTO = scelta_encoding["encoding"]
print(f"Codifica scelta: {ENCODING_SCELTO} (modello migliore: {scelta_encoding['modello']}, "
      f"RMSE {scelta_encoding['rmse_mean']:,.0f} ± {scelta_encoding['rmse_std']:,.0f} INR)")
migliori_per_encoding

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: motivazione della scelta; se le codifiche sono equivalenti, dirlo
# e spiegare che la scelta segue l'ordine di preferenza dichiarato>
#
# ### 7.3 L'errore al variare di alpha

# %%
fig = grafico_errore_vs_alpha([r for r in risultati_prezzo if r["encoding"] == ENCODING_SCELTO],
                              f"RMSE di CV al variare di alpha (codifica {ENCODING_SCELTO})")
plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: forma delle curve (piatta a sinistra = equivalente alla regressione
# lineare, sale a destra = troppa penalità); dove cade l'alpha scelto>
#
# ### 7.4 Come il Lasso seleziona le feature

# %%
fig = grafico_percorso_lasso(X_train, y_train, ENCODING_SCELTO,
                             f"Percorso dei coefficienti del Lasso (codifica {ENCODING_SCELTO})")
plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: quali feature vengono azzerate per prime (meno utili) e quali
# resistono più a lungo (più importanti)>
#
# ### 7.5 Quanti coefficienti restano diversi da zero

# %%
fig = grafico_barre_modelli(df_prezzo, "n_coef_nonzero", "Coefficienti ≠ 0",
                            "Numero di coefficienti diversi da zero per codifica e modello")
plt.show()

# %% [markdown]
# ### 7.6 Coefficienti a confronto: Ridge, Lasso, Elastic Net

# %%
coef_confronto = pd.concat({
    nome: tabella_effetti(trova_risultato(risultati_prezzo, ENCODING_SCELTO, nome, "prezzo")["search"].best_estimator_)
          .set_index("feature")["coef_std"]
    for nome in NOMI_MODELLI
}, axis=1).dropna()
coef_confronto

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: Ridge riduce ma non azzera; Lasso/ElasticNet azzerano X
# coefficienti; quali feature hanno i coefficienti più grandi in valore assoluto e cosa significa
# per la lettura del fenomeno>
#
# ### 7.7 One-hot vs ordinal: cosa cambia nei coefficienti di `furnishingstatus`

# %%
pd.concat({
    encoding: tabella_effetti(trova_risultato(risultati_prezzo, encoding, "Lasso", "prezzo")["search"].best_estimator_)
              .set_index("feature").filter(regex="arred|furnish", axis=0)[["coef_std", "effetto_inr"]]
    for encoding in ENCODINGS
}).rename_axis(["codifica", "feature"])

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: l'ordinal corretto impone salti uguali tra livelli, il one-hot stima
# due effetti separati (sono simili? il Lasso ne azzera uno?); la codifica ingenua dà un coefficiente
# più piccolo/inaffidabile perché l'ordine è sbagliato>
```

- [ ] **Step 3: Rigenerare, eseguire, scrivere le osservazioni**

Comandi come Task 5 Step 3. Se `df_prezzo["alpha_al_bordo"]` contiene `True`, allargare la relativa griglia in `GRIGLIE_ALPHA` di due decadi sul lato interessato, rieseguire `pytest tests -q`, rigenerare ed eseguire. Sostituire tutti i segnaposto; rigenerare ed eseguire.
Verifica: `grep -n "da scrivere" notebooks/01_confronto_modelli_regolarizzati.py` → nessun risultato; nessun `alpha_al_bordo == True` nell'output.

- [ ] **Step 4: Commit**

```bash
git add notebooks
git commit -m "Notebook 01: modelli regolarizzati per codifica, confronto e scelta"
```

---

### Task 8: Sezioni 8–9 – log del prezzo e confronto

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`

**Interfaces:**
- Consumes: `ENCODING_SCELTO`, `risultati_prezzo`, `df_prezzo`, funzioni di supporto.
- Produces: `risultati_log`, `df_log`, `scelta_prezzo: Series`, `scelta_log: Series`, `scelta_finale: Series`, `risultati_tutti: list[dict]`.

- [ ] **Step 1: Aggiungere le sezioni 8 e 9**

```python
# %% [markdown]
# ## 8. Passaggio al logaritmo del prezzo
# Nella sezione 4 abbiamo visto che il prezzo è asimmetrico: poche case molto costose "tirano" il modello.
# Addestrando sul **logaritmo del prezzo**:
# - la distribuzione diventa più simmetrica;
# - gli effetti diventano **percentuali** (es. "l'aria condizionata aumenta il prezzo del X%") invece che
#   in rupie fisse – un'ipotesi spesso più realistica, perché lo stesso optional vale di più su una casa costosa.
#
# Le predizioni vengono riconvertite in rupie prima di calcolare le metriche, quindi i numeri sono
# **direttamente confrontabili** con la sezione 6. Una nota tecnica: riconvertire con l'esponenziale stima
# la *mediana* del prezzo, non la media, e tende a sottostimare leggermente.

# %%
risultati_log = [addestra_configurazione(X_train, y_train, ENCODING_SCELTO, nome, log_target=True)
                 for nome in NOMI_MODELLI]
df_log = tabella_risultati(risultati_log)
df_log

# %% [markdown]
# **Cosa osserviamo** – <da scrivere>
#
# ## 9. Prezzo originale vs logaritmo
# ### 9.1 Metriche

# %%
df_confronto = pd.concat([df_prezzo[df_prezzo["encoding"] == ENCODING_SCELTO], df_log], ignore_index=True)
fig = grafico_barre_modelli(df_confronto, "rmse_mean", "RMSE medio di CV (INR)",
                            "RMSE di cross-validation: prezzo originale vs log(prezzo)", x="target",
                            colonna_errore="rmse_std")
plt.show()
fig = grafico_barre_modelli(df_confronto, "mae_mean", "MAE medio di CV (INR)",
                            "MAE di cross-validation: prezzo originale vs log(prezzo)", x="target")
plt.show()
df_confronto[["target", "modello", "rmse_mean", "rmse_std", "mae_mean", "r2_mean", "n_coef_nonzero"]]

# %% [markdown]
# ### 9.2 Residui: il log riduce gli errori sistematici sulle case costose?
# Per guardare i residui senza toccare il test set usiamo predizioni **fuori campione** sul training: ogni
# casa è predetta da un modello addestrato senza di lei.

# %%
scelta_prezzo = scegli_migliore(df_prezzo[df_prezzo["encoding"] == ENCODING_SCELTO], "modello", ORDINE_MODELLI)
scelta_log = scegli_migliore(df_log, "modello", ORDINE_MODELLI)
risultati_tutti = risultati_prezzo + risultati_log
stimatore_prezzo = trova_risultato(risultati_tutti, ENCODING_SCELTO, scelta_prezzo["modello"], "prezzo")["search"].best_estimator_
stimatore_log = trova_risultato(risultati_tutti, ENCODING_SCELTO, scelta_log["modello"], "log(prezzo)")["search"].best_estimator_
oof_prezzo = predizioni_oof(stimatore_prezzo, X_train, y_train)
oof_log = predizioni_oof(stimatore_log, X_train, y_train)

fig = grafico_residui_vs_predetti({f"prezzo – {scelta_prezzo['modello']}": (y_train, oof_prezzo),
                                   f"log(prezzo) – {scelta_log['modello']}": (y_train, oof_log)},
                                  "Residui fuori campione sul training")
plt.show()
pd.concat({"prezzo": errore_per_fascia(y_train, oof_prezzo).set_index("fascia"),
           "log(prezzo)": errore_per_fascia(y_train, oof_log).set_index("fascia")}, axis=1)

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: forma a imbuto dei residui? residuo medio sulla fascia più costosa
# (positivo = sottostima sistematica) nei due approcci>
#
# ### 9.3 Scelta finale (sul solo training)

# %%
scelta_finale = scegli_migliore(pd.DataFrame([scelta_prezzo, scelta_log]).reset_index(drop=True),
                               "target", ORDINE_TARGET)
print(f"Configurazione finale: {scelta_finale['modello']} su {scelta_finale['target']}, "
      f"codifica {ENCODING_SCELTO}, parametri {scelta_finale['best_params']}")

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: motivazione>
```

- [ ] **Step 2: Rigenerare, eseguire, scrivere le osservazioni**

Comandi come Task 5 Step 3. Controllare `alpha_al_bordo` in `df_log` (stessa procedura del Task 7). Sostituire i segnaposto; rigenerare ed eseguire.
Verifica: `grep -n "da scrivere" notebooks/01_confronto_modelli_regolarizzati.py` → nessun risultato.

- [ ] **Step 3: Commit**

```bash
git add notebooks
git commit -m "Notebook 01: target logaritmico e confronto con il prezzo originale"
```

---

### Task 9: Sezioni 10–11 – valutazione finale, conclusioni, verifica Colab

**Files:**
- Modify: `notebooks/01_confronto_modelli_regolarizzati.py`, `notebooks/01_confronto_modelli_regolarizzati.ipynb`

**Interfaces:**
- Consumes: `scelta_prezzo`, `scelta_log`, `scelta_finale`, `risultati_tutti`, `ENCODING_SCELTO`, `X_test`, `y_test`, `baseline`.

- [ ] **Step 1: Aggiungere le sezioni 10 e 11**

```python
# %% [markdown]
# ## 10. Valutazione finale sul test set
# Ora, e solo ora, usiamo il test set. Valutiamo i due finalisti (miglior modello sul prezzo e sul log)
# e, come riferimento, la regressione lineare non regolarizzata riaddestrata su tutto il training.
# Le scelte sono già state fatte: questi numeri servono a stimare come si comporterà il modello su case nuove.

# %%
finalisti = {
    "Lineare (baseline) – prezzo": costruisci_pipeline(ENCODING_SCELTO, "Lineare").fit(X_train, y_train),
    f"{scelta_prezzo['modello']} – prezzo": stimatore_prezzo,
    f"{scelta_log['modello']} – log(prezzo)": stimatore_log,
}
risultati_test = pd.DataFrame({nome: metriche(y_test, est.predict(X_test)) for nome, est in finalisti.items()}).T
risultati_test["rmse CV (training)"] = [
    trova_risultato(risultati_tutti, ENCODING_SCELTO, "Lineare", "prezzo")["rmse_mean"],
    scelta_prezzo["rmse_mean"], scelta_log["rmse_mean"]]
risultati_test

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: RMSE test vs RMSE CV (coerenti = la CV era affidabile);
# guadagno rispetto alla baseline>
#
# ### 10.1 Learning curve

# %%
for nome in list(finalisti)[1:]:
    fig = grafico_learning_curve(finalisti[nome], X_train, y_train, f"Learning curve – {nome}")
    plt.show()

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: le curve convergono? divario residuo? più dati aiuterebbero?>
#
# ### 10.2 Residui del modello finale sul test set

# %%
nome_finale = f"{scelta_finale['modello']} – {scelta_finale['target']}"
stimatore_finale = finalisti[nome_finale]
y_pred_test = stimatore_finale.predict(X_test)
fig = grafico_residui(y_test, y_pred_test, f"Diagnostica dei residui sul test set – {nome_finale}")
plt.show()
fig = grafico_predetto_vs_reale(y_test, y_pred_test, f"Predetto vs reale sul test set – {nome_finale}")
plt.show()
errore_per_fascia(y_test, y_pred_test)

# %% [markdown]
# **Cosa osserviamo** – <da scrivere: simmetria dei residui, eventuale sottostima delle case più costose
# (fascia alta con residuo medio positivo), cosa implica per i clienti>
#
# ## 11. Conclusioni e raccomandazione
# ### 11.1 Cosa pesa sul prezzo

# %%
log_finale = scelta_finale["target"] == "log(prezzo)"
effetti = tabella_effetti(stimatore_finale, log_target=log_finale,
                          prezzo_riferimento=float(y_train.median()) if log_finale else None)
print(f"Tasso di cambio usato: 1 € = {INR_PER_EUR} INR ({TASSO_FONTE})")
if log_finale:
    print(f"Effetti in euro calcolati su una casa di prezzo mediano ({y_train.median():,.0f} INR)")
fig = grafico_effetti(effetti, "effetto_eur", "Effetto sul prezzo (€)",
                      f"Effetto di ciascuna caratteristica sul prezzo – {nome_finale}")
plt.show()
effetti

# %% [markdown]
# ### 11.2 Sintesi
# <da scrivere, in elenco puntato, con i numeri effettivi:
# - overfitting della baseline: cosa abbiamo trovato davvero;
# - codifica: differenze misurate tra ordinal ingenuo, ordinal corretto e one-hot e scelta fatta;
# - modelli: RMSE/MAE/R² di CV e test dei finalisti, quanti coefficienti ≠ 0 per modello;
# - log(prezzo): effetto su errori e residui per fascia;
# - le 3–5 caratteristiche più influenti con il loro effetto in € (o %).>
#
# ### 11.3 Raccomandazione operativa
# <da scrivere: quale modello adottare nel motore di valutazione, con quale codifica e target, a quali
# condizioni (es. ricalibrare alpha con dati nuovi, segnalare ai clienti minore affidabilità nella fascia
# di prezzo più alta se confermato dai residui)>
#
# ### 11.4 Limiti
# <da scrivere: 545 case di un solo mercato (India); poche feature (nessuna posizione precisa, età
# dell'immobile); stime di CV leggermente ottimistiche perché alpha è scelto sulla stessa CV; la
# retrotrasformazione del log stima la mediana. Rimandi agli approfondimenti possibili.>
```

- [ ] **Step 2: Rigenerare, eseguire, scrivere conclusioni**

Comandi come Task 5 Step 3. Sostituire tutti i segnaposto con testo basato sugli output effettivi. Rigenerare ed eseguire.
Verifica: `grep -n "da scrivere" notebooks/01_confronto_modelli_regolarizzati.py` → nessun risultato.

- [ ] **Step 3: Verifiche finali**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests -q
grep -nE "C:\\\\|/Users/|\.venv|%pip|import nb_loader" notebooks/01_confronto_modelli_regolarizzati.py
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_confronto_modelli_regolarizzati.ipynb
.venv/Scripts/python.exe -c "import nbformat; nb = nbformat.read('notebooks/01_confronto_modelli_regolarizzati.ipynb', 4); errs = [o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error']; print('celle con errore:', len(errs))"
```
Expected: test PASS; il `grep` non trova nulla; esecuzione senza errori; `celle con errore: 0`.

- [ ] **Step 4: Commit**

```bash
git add notebooks
git commit -m "Notebook 01: valutazione finale, conclusioni e raccomandazione"
```

- [ ] **Step 5: Verifica manuale su Colab (utente)**

Chiedere all'utente di caricare `notebooks/01_confronto_modelli_regolarizzati.ipynb` su Colab ed eseguire *Runtime → Esegui tutto*; riportare eventuali errori (es. versioni di libreria) e correggerli.
