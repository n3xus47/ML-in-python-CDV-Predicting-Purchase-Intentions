import pytest
import pandas as pd
import numpy as np
from ml_classes import FeatureEngineer


@pytest.fixture
def fe_data():
    """Dane testowe dla FeatureEngineera."""
    return pd.DataFrame({
        'Administrative': [1, 2],
        'Informational': [1, 1],
        'ProductRelated': [3, 7],
        'Administrative_Duration': [10, 20],
        'Informational_Duration': [5, 5],
        'ProductRelated_Duration': [15, 25]
    })


def test_feature_creation(fe_data):
    fe = FeatureEngineer()
    df_new = fe.create_interaction_features(fe_data)

    # Sprawdzamy czy TotalPages to suma (1+1+3 = 5) - tu są liczby całkowite, więc == jest OK
    assert df_new['TotalPages'].iloc[0] == 5

    # Sprawdzamy czy TotalDuration to suma (10+5+15 = 30)
    assert df_new['TotalDuration'].iloc[0] == 30

    # Używamy pytest.approx, aby uwzględnić bezpieczny margines błędu (1e-6)
    # 30 / 5.000001 = 5.9999988...
    assert df_new['AvgPageDuration'].iloc[0] == pytest.approx(6.0, rel=1e-5)


def test_select_features(fe_data):
    """Dodatkowy test sprawdzający selekcję cech."""
    fe = FeatureEngineer()
    y = pd.Series([0, 1])

    # Test metody korelacyjnej
    X_selected, selected_cols = fe.select_features(fe_data, y, method='correlation', threshold=0.1)

    assert isinstance(X_selected, pd.DataFrame)
    assert len(selected_cols) > 0