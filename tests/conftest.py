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
