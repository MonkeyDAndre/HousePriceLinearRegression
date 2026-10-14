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
