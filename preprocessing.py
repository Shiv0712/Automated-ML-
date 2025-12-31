import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(df, selected_columns, target_column=None, categorical_columns=None, model_folder='my_models'):
    df_selected = df[selected_columns]

    # Optionally cast known categorical columns to 'category' dtype
    if categorical_columns is not None:
        for col in categorical_columns:
            if col in df_selected.columns:
                df_selected[col] = df_selected[col].astype('category')

    X = df_selected.copy()

    # Encode target if available (always label encode if categorical)
    y = None
    label_encoder = None
    if target_column:
        target_series = df[target_column]
        if target_series.dtype == 'object' or str(target_series.dtype) == 'category':
            le = LabelEncoder()
            y = le.fit_transform(target_series.astype(str))
            label_encoder = le
            # Save the label encoder for later use
            encoder_path = os.path.join(model_folder, f'label_encoder_{target_column}.pkl')
            joblib.dump(le, encoder_path)
        else:
            y = target_series.values  # No scaling for regression, just use raw values

    # Split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        return X_train, X_test, None, None
