import pytest
import pandas as pd
from sklearn.datasets import make_classification
from ml_classes import ModelTrainer


# Fixtura generująca dane - musi być tutaj, skoro nie używamy conftest.py
@pytest.fixture
def dummy_split():
    """Generuje syntetyczne dane do testów (100 próbek, 5 cech)."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
    return X, y


def test_training_and_evaluation(dummy_split):
    """Testuje pełny proces: trenowanie modelu i obliczanie metryk."""
    X, y = dummy_split
    trainer = ModelTrainer()

    # 1. Test trenowania
    model = trainer.train_model(X, y, model_type='logistic')
    assert 'logistic' in trainer.models

    # 2. Test ewaluacji (czy zwraca słownik z poprawnymi kluczami)
    metrics = trainer.evaluate_model(model, X, y)

    expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in expected_metrics:
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1  # Metryki muszą być w przedziale 0-1


def test_compare_models(dummy_split):
    """Testuje funkcję porównywania wielu modeli."""
    X, y = dummy_split
    trainer = ModelTrainer()

    # Trenujemy dwa różne modele
    m1 = trainer.train_model(X, y, model_type='logistic')
    m2 = trainer.train_model(X, y, model_type='random_forest')

    models_to_compare = {'LogReg': m1, 'RF': m2}
    results_df = trainer.compare_models(models_to_compare, X, y)

    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == 2
    assert 'accuracy' in results_df.columns