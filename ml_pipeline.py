from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import TargetEncoder
import pandas as pd
import numpy as np
import sys

# Custom transformer for column selection
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]

# Helper to determine if a model supports native categorical features
TREE_MODELS_NATIVE_CAT = [
    'HistGradientBoostingClassifier', 'HistGradientBoostingRegressor',
    'LGBMClassifier', 'LGBMRegressor', 'CatBoostClassifier', 'CatBoostRegressor'
]

def get_model_name(model):
    return type(model).__name__

def choose_encoder(col, n_unique, model_name, y_available):
    # Thresholds can be tuned
    LOW_CARDINALITY = 10
    if model_name in TREE_MODELS_NATIVE_CAT:
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif n_unique <= LOW_CARDINALITY:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif y_available:
        return TargetEncoder()
    else:
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

class AutoEncoderSelector(BaseEstimator, TransformerMixin):
    printed = False  # Class variable to ensure printing only once per process
    def __init__(self, y=None, model=None):
        self.y = y
        self.model = model
        self.encoders = {}
        self.cat_cols = []
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        # if not AutoEncoderSelector.printed:
        #     print("[AutoML Encoding] --- Fitting AutoEncoderSelector ---")
        #     print(f"[AutoML Encoding] X.dtypes:\n{X.dtypes}")
        #     print(f"[AutoML Encoding] Detected categorical columns: {self.cat_cols}")
        #     sys.stdout.flush()
        #     AutoEncoderSelector.printed = True
        model_name = get_model_name(self.model) if self.model is not None else ''
        for col in self.cat_cols:
            n_unique = X[col].nunique(dropna=True)
            y_available = self.y is not None and len(self.y) == len(X)
            encoder = choose_encoder(col, n_unique, model_name, y_available)
            encoder_name = type(encoder).__name__
            # print(f"[AutoML Encoding] Column '{col}': {encoder_name} (unique values: {n_unique})")
            # sys.stdout.flush()
            if isinstance(encoder, TargetEncoder):
                encoder.fit(X[[col]], self.y)
            else:
                encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self
    def transform(self, X):
        X = X.copy()
        for col, encoder in self.encoders.items():
            enc = encoder.transform(X[[col]])
            if hasattr(enc, 'toarray'):
                enc = enc.toarray()
            if enc.ndim == 1 or enc.shape[1] == 1:
                X[col] = enc.ravel()
            else:
                # For OHE, expand columns
                for i in range(enc.shape[1]):
                    X[f'{col}_enc_{i}'] = enc[:, i]
                X = X.drop(columns=[col])
        return X

# Custom transformer for correlated feature removal
class CorrelatedFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.to_drop_ = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        else:
            self.to_drop_ = []
        return self
    def transform(self, X):
        X = X.copy()
        if self.to_drop_:
            X = X.drop(columns=self.to_drop_, errors='ignore')
        return X

# Function to build the pipeline
def build_pipeline(selected_columns, model, apply_pca=False, remove_corr=False, y=None, sample_df=None):
    # Use sample_df to infer types
    if sample_df is not None:
        cat_cols = sample_df[selected_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = sample_df[selected_columns].select_dtypes(include=['number', 'float', 'int']).columns.tolist()
    else:
        cat_cols, num_cols = [], []

    steps = [('selector', ColumnSelector(selected_columns))]
    steps.append(('auto_encoder', AutoEncoderSelector(y=y, model=model)))
    if num_cols:
        steps.append(('scaler', StandardScaler()))
    if remove_corr:
        steps.append(('remove_corr', CorrelatedFeatureRemover()))
    if apply_pca:
        from sklearn.decomposition import PCA
        steps.append(('pca', PCA(n_components=2)))
    steps.append(('model', model))
    pipeline = Pipeline(steps)
    return pipeline 