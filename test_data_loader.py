import pytest
import pandas as pd
from ml_classes import DataLoader


@pytest.fixture
def loader_setup(tmp_path):
    """Przygotowuje loader i tymczasowy plik CSV."""
    path = tmp_path / "data.csv"
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df.to_csv(path, index=False)
    return DataLoader(), str(path)


def test_loader_success(loader_setup):
    loader, path = loader_setup
    df = loader.load_data(path)

    assert df is not None
    assert df.shape == (2, 2)

    info = loader.get_info()
    assert 'shape' in info
    assert info['shape'] == (2, 2)


def test_loader_file_not_found():
    loader = DataLoader()
    assert loader.load_data("non_existent.csv") is None