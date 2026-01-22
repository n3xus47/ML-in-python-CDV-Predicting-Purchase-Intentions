"""
Klasy OOP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from collections import Counter


def check_and_handle_imbalance(y_train, threshold=0.8):
    """
    Sprawdza niezbalansowanie klas i zwraca odpowiednie parametry
    
    """
    class_counts = Counter(y_train)
    if len(class_counts) != 2:
        return {}
    
    counts = list(class_counts.values())
    imbalance_ratio = min(counts) / max(counts)
    
    # jesli stosunek mniejszy niz threshold to klasy sa niezbalansowane
    if imbalance_ratio < threshold:
        return {'class_weight': 'balanced'}
    return {}


class DataLoader:
    """Klasa odpowiedzialna za wczytanie danych z pliku CSV"""
    
    def __init__(self):
        self.data = None
        self.path = None
    
    def load_data(self, path: str) -> pd.DataFrame:
        try:
            self.data = pd.read_csv(path)
            self.path = path
            print(f"Dane wczytane pomyślnie z: {path}")
            print(f"Kształt danych: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Błąd: Plik nie został znaleziony: {path}")
            return None
        except Exception as e:
            print(f"Błąd podczas wczytania danych: {e}")
            return None
    
    def get_info(self) -> dict:
        """
        Zwraca podstawowe informacje o zbiorze danych
        
        """
        if self.data is None:
            return {"error": "Dane nie zostały wczytane"}
        
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum()
        }
        return info


class DataPreprocessor:
    """Klasa odpowiedzialna za preprocessing danych"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Obsługuje brakujące wartości w danych.
        
       Strategia obsługi braków ('drop' lub 'mean')
                - 'drop': Usuwa wiersze z brakującymi wartościami
                - 'mean': Wypełnia średnią (tylko kolumny numeryczne)
                
        """
        df_processed = df.copy()
        
        if strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'mean':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        else:
            raise ValueError(f"Nieznana strategia: {strategy}. Wybierz: 'drop' lub 'mean'")
        
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, columns: list = None, method: str = 'label') -> pd.DataFrame:
        """
        Koduje zmienne kategoryczne na numeryczne.
        
        Metoda kodowania ('label' lub 'onehot')
                - 'label': Label Encoding (zamiana na liczby)
                - 'onehot': One-Hot Encoding (tworzenie kolumn binarnych)
                
        """
        df_encoded = df.copy()
        
        if columns is None:
            categorical_cols = df_encoded.select_dtypes(include=['object', 'bool']).columns.tolist()
        else:
            categorical_cols = columns
        
        if method == 'label':
            # zapisujemy encodery zeby moc pozniej zakodowac nowe dane
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
        else:
            raise ValueError(f"Nieznana metoda kodowania: {method}. Wybierz: 'label' lub 'onehot'")
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, columns: list = None, fit: bool = True) -> pd.DataFrame:
        """
        Normalizuje zmienne numeryczne używając StandardScaler (średnia=0, std=1).
        
        """
        df_normalized = df.copy()
        
        if columns is None:
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = columns
        
        if fit:
            # fit_transform dla danych treningowych - uczy parametry i transformuje
            df_normalized[numeric_cols] = self.scaler.fit_transform(df_normalized[numeric_cols])
            self.is_fitted = True
        else:
            # transform dla danych testowych - uzywa juz nauczonych parametrow
            if not self.is_fitted:
                raise ValueError("Scaler nie został dopasowany. Użyj fit=True dla danych treningowych.")
            df_normalized[numeric_cols] = self.scaler.transform(df_normalized[numeric_cols])
        
        return df_normalized
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str, 
                           normalize: bool = True, fit: bool = True) -> tuple:
        """
        Pełny pipeline preprocessingu z domyślnymi ustawieniami.
        
        Wykonuje kolejno: obsługę braków -> kodowanie kategorycznych -> normalizację.
        
        """
        df_processed = df.copy()
        df_processed = self.handle_missing_values(df_processed, strategy='drop')
        
        # oddzielamy target od features
        if target_col not in df_processed.columns:
            raise ValueError(f"Kolumna '{target_col}' nie istnieje w danych")
        y = df_processed[target_col].copy()
        X = df_processed.drop(columns=[target_col])
        
        # kodowanie kategorycznych przed normalizacja
        X = self.encode_categorical(X, method='label')
        
        if normalize:
            X = self.normalize_features(X, fit=fit)
        
        return (X, y)


class DataAnalyzer:
    """Klasa odpowiedzialna za analizę danych i wizualizacje"""
    
    def __init__(self):
        pass
    
    def descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zwraca statystyki opisowe
        
        """
        return df.describe()
    
    def correlation_analysis(self, df: pd.DataFrame, target: str) -> pd.Series:
        """
        Analiza korelacji zmiennych z targetem.
        
        """
        df_processed = df.copy()
        numeric_df = df_processed.select_dtypes(include=[np.number])
        
        # jesli target nie jest numeryczny to kodujemy go zeby policzyc korelacje
        if target not in numeric_df.columns:
            if target not in df_processed.columns:
                raise ValueError(f"Kolumna '{target}' nie istnieje w danych")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_processed[target] = le.fit_transform(df_processed[target].astype(str))
            numeric_df = df_processed.select_dtypes(include=[np.number])
        
        if target not in numeric_df.columns:
            raise ValueError(f"Kolumna '{target}' nie jest numeryczna i nie można jej zakodować")
        
        correlations = numeric_df.corr()[target].sort_values(ascending=False)
        return correlations
    
    def visualize_correlations(self, df: pd.DataFrame, figsize: tuple = (12, 10)):
        """
        Wizualizuje macierz korelacji
        
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Macierz korelacji')
        plt.tight_layout()
        plt.show()
    
    def class_balance_analysis(self, y: pd.Series) -> dict:
        """
        Analiza balansu klas
        
        """
        value_counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100
        
        analysis = {
            "counts": value_counts.to_dict(),
            "percentages": percentages.to_dict(),
            "is_balanced": (percentages.min() > 40) and (percentages.max() < 60)
        }
        
        plt.figure(figsize=(8, 5))
        value_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Rozkład klas docelowych')
        plt.xlabel('Klasa')
        plt.ylabel('Liczba obserwacji')
        plt.xticks(rotation=0)
        for i, v in enumerate(value_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        
        return analysis
    
    def visualize_distributions(self, df: pd.DataFrame, columns: list = None, 
                                metadata: dict = None, figsize: tuple = (15, 10)):
        """
        Wizualizuje rozkłady wybranych zmiennych numerycznych.
        
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        columns = [col for col in columns if col in df.columns]
        
        if not columns:
            print("Brak kolumn numerycznych do wizualizacji")
            return
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # matplotlib zwraca rozne typy w zaleznosci od liczby subplotow, trzeba to ujednolicic
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            ax = axes[idx]
            
            if 'Revenue' in df.columns and col != 'Revenue':
                for revenue_value in df['Revenue'].unique():
                    subset = df[df['Revenue'] == revenue_value][col]
                    label = 'Purchase' if revenue_value else 'No Purchase'
                    ax.hist(subset, alpha=0.6, label=label, bins=30)
                ax.legend()
            else:
                ax.hist(df[col].dropna(), bins=30, alpha=0.7, color='steelblue')
            
            title = col
            if metadata and col in metadata:
                title += f" ({metadata[col]})"
            ax.set_title(title)
            ax.set_xlabel(col)
            ax.set_ylabel('Częstość')
            ax.grid(alpha=0.3)
        
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()


class FeatureEngineer:
    """Klasa odpowiedzialna za feature engineering"""
    
    def __init__(self):
        self.feature_importance = None
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tworzy cechy interakcyjne
        
        """
        df_new = df.copy()
        
        if all(col in df_new.columns for col in ['Administrative', 'Informational', 'ProductRelated']):
            df_new['TotalPages'] = (df_new['Administrative'] + 
                                   df_new['Informational'] + 
                                   df_new['ProductRelated'])
        
        if all(col in df_new.columns for col in ['Administrative_Duration', 
                                                  'Informational_Duration', 
                                                  'ProductRelated_Duration']):
            df_new['TotalDuration'] = (df_new['Administrative_Duration'] + 
                                       df_new['Informational_Duration'] + 
                                       df_new['ProductRelated_Duration'])
        
        if 'TotalDuration' in df_new.columns and 'TotalPages' in df_new.columns:
            # 1e-6 zeby uniknac dzielenia przez zero
            df_new['AvgPageDuration'] = df_new['TotalDuration'] / (df_new['TotalPages'] + 1e-6)
        
        if all(col in df_new.columns for col in ['BounceRates', 'ExitRates']):
            df_new['BounceExitRatio'] = df_new['BounceRates'] / (df_new['ExitRates'] + 1e-6)
        
        if 'ProductRelated' in df_new.columns and 'TotalPages' in df_new.columns:
            df_new['ProductRelatedRatio'] = df_new['ProductRelated'] / (df_new['TotalPages'] + 1e-6)
        
        return df_new
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'importance', threshold: float = 0.01) -> tuple:
        """
        Selekcja zmiennych
        
        """
        if method == 'correlation':
            # wybieramy cechy z korelacja powyzej progu
            correlations = X.corrwith(y).abs()
            selected_features = correlations[correlations >= threshold].index.tolist()
            X_selected = X[selected_features]
        
        elif method == 'importance':
            # rf uczy sie na wszystkich cechach i zwraca ich waznosc
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            self.feature_importance = feature_importance.sort_values(ascending=False)
            
            selected_features = feature_importance[feature_importance >= threshold].index.tolist()
            X_selected = X[selected_features]
    
        else:
            X_selected = X
            selected_features = X.columns.tolist()
        
        return X_selected, selected_features


class ModelTrainer:
    """Klasa odpowiedzialna za trenowanie modeli"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_model(self, X_train, y_train, model_type: str, 
                   handle_imbalance: bool = True, **kwargs):
        """
        Trenuje model klasyfikacji.
    
        """
        # automatyczna obsługa niezbalansowanych klas
        if handle_imbalance:
            imbalance_params = check_and_handle_imbalance(y_train)
            if imbalance_params:
                class_counts = Counter(y_train)
                print(f"Wykryto niezbalansowanie klas: {class_counts}")
                for key, value in imbalance_params.items():
                    if key not in kwargs:
                        kwargs[key] = value
                        print(f"Stosowanie {key}='{value}'")
        
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000, **kwargs)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
        elif model_type == 'svm':
            model = SVC(random_state=42, probability=True, **kwargs)
        else:
            raise ValueError(f"Nieznany typ modelu: {model_type}")
        
        model.fit(X_train, y_train)
        self.models[model_type] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test) -> dict:
        """
        Ewaluuje model i zwraca metryki wydajności.
         
        """
        y_pred = model.predict(X_test)
        
        # bierzemy prawdopodobienstwo do ROC AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def compare_models(self, models: dict, X_test, y_test) -> pd.DataFrame:
        """
        Porównuje wiele modeli i zwraca DataFrame z metrykami.
        
        """
        results = []
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics['model'] = name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('model')
        
        return results_df


class HyperparameterTuner:
    """Klasa odpowiedzialna za optymalizację hiperparametrów"""
    
    def __init__(self):
        self.best_params = {}
        self.best_scores = {}
    
    def _create_pipeline(self, model):
        """Tworzy Pipeline ze StandardScaler i modelem"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    def grid_search(self, model, param_grid: dict, X_train, y_train, 
                   cv: int = 5, scoring: str = 'f1', n_jobs: int = -1,
                   handle_imbalance: bool = True):
        """
        Grid Search dla optymalizacji hiperparametrów z użyciem Pipeline.
        
        """
        # pipeline wymaga prefixu 'model__' dla parametrow wewnetrznego modelu
        if handle_imbalance:
            imbalance_params = check_and_handle_imbalance(y_train)
            if imbalance_params:
                if 'model__class_weight' not in param_grid:
                    if hasattr(model, 'set_params'):
                        model.set_params(class_weight='balanced')
                    print("Dodano class_weight='balanced' do modelu w Pipeline")
        
        pipeline = self._create_pipeline(model)
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params[type(model).__name__] = grid_search.best_params_
        self.best_scores[type(model).__name__] = grid_search.best_score_
        
        print(f"Najlepsze parametry: {grid_search.best_params_}")
        print(f"Najlepszy wynik CV: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def random_search(self, model, param_distributions: dict, X_train, y_train,
                     n_iter: int = 50, cv: int = 5, scoring: str = 'f1', n_jobs: int = -1):
        """
        Random Search dla optymalizacji hiperparametrów z użyciem Pipeline.
        
        """
        pipeline = self._create_pipeline(model)
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params[type(model).__name__] = random_search.best_params_
        self.best_scores[type(model).__name__] = random_search.best_score_
        
        print(f"Najlepsze parametry: {random_search.best_params_}")
        print(f"Najlepszy wynik CV: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
