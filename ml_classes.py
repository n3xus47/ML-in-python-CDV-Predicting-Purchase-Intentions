import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from typing import Any
""""klasy"""

class DataLoader:

    def __init__(self):
        self.data = None
        self.path = None

    def load_data(self, path: str):

        try:
            self.data = pd.read_csv(path)
            self.path = path
            print("Data loaded")
            print(f"Data shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"File not found at {path}")
            return None
        except Exception as e:
            print(f"Error while loading data: {e}")
            return None

    def get_info(self):
        if self.data is None:
            return {"error": "Data not loaded"}

        info = {
            "shape": self.data.shape,
            "columns": list[Any](self.data.columns),
            "dtypes": (self.data.dtypes).to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
        }
        return info



