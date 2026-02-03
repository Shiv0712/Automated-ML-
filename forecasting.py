import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import logging

def create_time_series_features(df, target_col, date_col, frequency):
    """
    Generates time series features (Lags, Rolling, Date parts).
    """
    df = df.copy()
    
    # 1. Date Conversion & Sorting
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
    except Exception as e:
        raise ValueError(f"Could not convert '{date_col}' to datetime: {e}")

    # Set index for time-based operations (optional but good for resampling if needed)
    # df.set_index(date_col, inplace=True) 
    
    # 2. Extract Date Features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    
    if frequency in ['hourly', '15min']:
        df['hour'] = df[date_col].dt.hour
        df['minute'] = df[date_col].dt.minute

    # 3. Lag Features (Target t-1, t-2, t-3)
    # Note: Lags introduce NaNs at the beginning
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # 4. Rolling Window Features
    for window in [3, 7]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()

    # Drop NaNs created by shifting/rolling
    df = df.dropna()
    
    return df

def get_base_model(algorithm):
    if algorithm == "Random Forest":
        return RandomForestRegressor(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif algorithm == "XGBoost":
        return XGBRegressor(objective='reg:squarederror'), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif algorithm == "Linear Regression":
        return LinearRegression(), {}
    elif algorithm == "Ridge":
        return Ridge(), {'alpha': [0.1, 1.0, 10.0]}
    
    return None, {}


def train_time_series_model(X_train, y_train, algorithm, X_test=None, y_test=None):
    """
    Trains a time series model using TimeSeriesSplit validation.
    """
    print(f"Training Time Series Model: {algorithm}")
    
    model, param_grid = get_base_model(algorithm)
    
    if model is None:
        return {
            "algorithm": algorithm,
            "error": "Model not supported",
            "success": False
        }, {}

    # TimeSeriesSplit for temporal validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    if param_grid:
        try:
            search = RandomizedSearchCV(
                model, param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error', 
                n_iter=10, n_jobs=-1, random_state=42
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
        except Exception as e:
            print(f"Error during Grid Search: {e}")
            model.fit(X_train, y_train)
            best_model = model
            best_params = "Fallback (Error)"
    else:
        model.fit(X_train, y_train)
        best_model = model
        best_params = "Default"

    # Evaluation
    results = {
        "algorithm": algorithm,
        "best_params": best_params,
        "success": True
    }
    
    # Evaluate on Train
    try:
        y_pred_train = best_model.predict(X_train)
        results["train_mape"] = round(mean_absolute_percentage_error(y_train, y_pred_train), 4)

        # Evaluate on Test (if provided)
        if X_test is not None and y_test is not None:
            y_pred_test = best_model.predict(X_test)
            results["mse"] = round(mean_squared_error(y_test, y_pred_test), 4)
            results["r2"] = round(r2_score(y_test, y_pred_test), 4)
            results["mape"] = round(mean_absolute_percentage_error(y_test, y_pred_test), 4)
            results["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        results["error"] = str(e)
    
    return [results], {algorithm: best_model}
