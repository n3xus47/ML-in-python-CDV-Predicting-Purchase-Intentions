import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ml_classes import HyperparameterTuner


# Definicja brakującej fixtury
@pytest.fixture
def dummy_split():
    """Generuje dane do testów modeli."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(y)
    return X, y


# Test gridsearch
def test_grid_search_flow(dummy_split):
    X, y = dummy_split
    tuner = HyperparameterTuner()
    model = LogisticRegression()

    # min siatka parametrów (szybki test)
    param_grid = {'model__C': [0.1, 1.0]}

    # uruchomienie tunera
    best_model = tuner.grid_search(model, param_grid, X, y, cv=2)

    # sprawdzanie wynikow
    assert best_model is not None
    assert 'LogisticRegression' in tuner.best_params

    assert tuner.best_scores['LogisticRegression'] > 0
