import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from typing import Any
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder


def check_and_handle_imbalance(y_train, threshold=0.8):
    class_counts = Counter[Any](y_train)
    if len(class_counts) != 2:
        return {}

    counts = list[int](class_counts.values())
    imbalance_ratio = min(counts) / max(counts)

    if imbalance_ratio < threshold:
        return {'class_weight': 'balanced'}
    return {}


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


class DataPreprocessor:

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop'):

        df_processed = df.copy()

        if strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'mean':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        else:
            raise ValueError(f"unknown strategy:{strategy}. select from drop or mean")

        return df_processed

    def encode_categorical(self, df: pd.DataFrame, columns: list = None, method: str = 'label'):

        df_encoded = df.copy()

        if columns is None:
            categorical_cols = df_encoded.select_dtypes(include=['object', 'bool']).columns.tolist()
        else:
            categorical_cols = columns

        if method == 'label':
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
        else:
            raise ValueError(f"unknown method:{method}. Select from label or onehot")

        return df_encoded

    def normalize_features(self, df: pd.DataFrame, columns: list = None, fit: bool = True):

        df_normalized = df.copy()

        if columns is None:
            numerical_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numerical_cols = columns

        if fit:
            # Poniższa linia generuje błąd w main.ipynb, ale zostawiam ją zgodnie z prośbą
            df_normalized[numerical_cols] = self.scaler.fit_transform(df_normalized[numerical_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Use fit=True for training data")
            df_normalized[numerical_cols] = self.scaler.transform(df_normalized[numerical_cols])

        return df_normalized

    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str,
                            normalize: bool = True, fit: bool = True):

        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed, strategy='drop')

        if target_col not in df_processed.columns:
            raise ValueError(f"Target_col:{target_col} not in dataframe")
        y = df_processed[target_col].copy()
        X = df_processed.drop(columns=[target_col])

        X = self.encode_categorical(X, method='label')

        if normalize:
            X = self.normalize_features(X, fit=fit)

        return (X, y)

class DataAnalyzer:
    def __init__(self):
        pass

    def descriptive_statistics(self, df: pd.DataFrame):

        return df.describe()
    def correlation_analysis(self, df: pd.DataFrame, target: str):
        df_processed = df.copy()
        numeric_df = df_processed.select_dtypes(include=[np.number])

        if target not in numeric_df.columns:
            if target not in df_processed.columns:
                raise ValueError(f"Target Column:{target} not in dataframe")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_processed[target] = le.fit_transform(df_processed[target].astype(str))
            numeric_df = df_processed.select_dtypes(include=[np.number])

        if target not in numeric_df.columns:
            raise ValueError(f"Target Column:{target} is not numeric and cannot be encoded")

        correlations = numeric_df.corr()[target].sort_values(ascending=False)
        return correlations
    def visualise_correlations(self, df: pd.DataFrame, figsize: tuple = (12, 10)):

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def class_balance_analysis(self, y: pd.Series):

        value_counts = y.value_counts()
        percentages = y.value_counts(normalize=True) *100

        analysis = {
            "counts": value_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "is_balanced": (percentages.min()>40) and (percentages.max()<60)
        }

        plt.figure(figsize=(8, 5))
        value_counts.plot(kind='bar', color=['skyblue','salmon'])
        plt.title('Target class distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of observations')
        plt.xticks(rotation=0)
        for i, v in enumerate(value_counts.values):
            plt.text(i, v , str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

        return analysis

    def visualize_distributions(self, df: pd.DataFrame,columns:list = None,
                               metadata:dict = None,  figsize: tuple = (15, 10)):

        if columns is None:
            colums = df.select_dtypes(include=[np.number]).columns.tolist()

        columns = [col for col in columns if col in df.columns]

        if not columns:
            print("No numeric columns for visualization")
            return

        n_cols = 3
        n_rows = (len(columns)+n_cols -1 )//n_cols

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, col in enumerate(columns):
            ax=axes[idx]

            if 'Revenue' in df.columns and col != 'Revenue':
                for revenue_value in df['Revenue'].unique():
                    subset = df[df['Revenue'] == revenue_value][col]
                    label = 'Purchase' if revenue_value else 'No Purchase'
                    ax.hist(subset,alpha=0.6, label=label, bins=30)
                ax.legend()
            else:
                ax.hist(df[col].dropna(), alpha=0.7, bins=30, color='steelblue')

            title=col
            if metadata and col in metadata:
                title += f"({metadata[col]})"
            ax.set_title(title)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)

        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()











