import pytest
import pandas as pd
import numpy as np
from ml_classes import DataPreprocessor


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


@pytest.fixture
def raw_data():
    """Dane z brakami i kategoriami."""
    return pd.DataFrame({
        'num': [1.0, np.nan, 3.0],
        'cat': ['A', 'B', 'A'],
        'target': [0, 1, 0]
    })


def test_handle_missing_values(preprocessor, raw_data):
    # Test strategii 'mean'
    df_mean = preprocessor.handle_missing_values(raw_data, strategy='mean')
    assert df_mean['num'].isnull().sum() == 0
    assert df_mean['num'].iloc[1] == 2.0  # średnia z 1 i 3


def test_encode_categorical(preprocessor, raw_data):
    # Test Label Encoding
    df_encoded = preprocessor.encode_categorical(raw_data, method='label')
    assert np.issubdtype(df_encoded['cat'].dtype, np.integer)


def test_preprocess_pipeline(preprocessor, raw_data):
    # Pełny proces: drop NaN -> label encode -> split X,y
    X, y = preprocessor.preprocess_pipeline(raw_data, target_col='target')

    # Po 'drop' zostaną 2 wiersze (bo jeden miał NaN)
    assert len(X) == 2
    assert 'target' not in X.columns
    assert len(y) == 2
