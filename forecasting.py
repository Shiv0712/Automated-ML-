import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import logging

# Set matplotlib backend to non-GUI before any plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_time_series_features(df, target_col, date_col, frequency):
    """Generates time series features (Lags, Rolling, Date parts).

    Args:
        df: pandas DataFrame containing the series.
        target_col: name of the column to forecast.
        date_col: name of the datetime column.
        frequency: one of ['15min', '30min', 'hourly', 'daily'].

    Returns:
        DataFrame with new features and no NaNs (drops initial rows).
    """
    df = df.copy()

    # 1. Date Conversion & Sorting
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    except Exception as e:
        raise ValueError(f"Could not convert '{date_col}' to datetime: {e}")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"Date column '{date_col}' is not a datetime dtype after conversion.")

    df = df.sort_values(by=date_col)

    # 2. Extract Date Features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter

    if frequency in ['hourly', '15min', '30min']:
        df['hour'] = df[date_col].dt.hour
        df['minute'] = df[date_col].dt.minute

    # 3. Lag Features (Target t-1, t-2, t-3, t-7)
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


def get_forecast_steps_for_next_day(frequency):
    """Return number of steps to forecast for the next day based on frequency."""
    if frequency == '15min':
        return 24 * 4
    if frequency == '30min':
        return 24 * 2
    if frequency == 'hourly':
        return 24
    # Default to 1 day of daily frequency
    return 1


def generate_forecast(df, model, date_col, target_col, frequency, steps=None):
    """Generate forecast values for the next `steps` periods.

    This function creates lag/rolling features iteratively, so each predicted value
    is used to generate the next prediction.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    df = df.sort_values(by=date_col)

    if steps is None:
        steps = get_forecast_steps_for_next_day(frequency)

    df_features = create_time_series_features(df, target_col, date_col, frequency)
    working = df_features.copy()

    last_timestamp = df[date_col].max()
    freq_map = {
        '15min': '15min',
        '30min': '30min',
        'hourly': 'H',
        'daily': 'D'
    }

    if frequency not in freq_map:
        raise ValueError(f"Unsupported frequency: {frequency}")

    period = pd.Timedelta(freq_map[frequency])
    forecast_rows = []

    for _ in range(steps):
        next_time = last_timestamp + period

        temp = {
            date_col: next_time,
            target_col: np.nan
        }
        temp_df = pd.DataFrame([temp])
        temp_df = pd.concat([working, temp_df], ignore_index=True, sort=False)

        temp_features = create_time_series_features(temp_df, target_col, date_col, frequency)
        last_row = temp_features.tail(1).copy()

        X_pred = last_row.drop(columns=[target_col, date_col])
        y_pred = model.predict(X_pred)[0]

        forecast_rows.append({
            date_col: next_time,
            target_col: y_pred
        })

        new_row = last_row.copy()
        new_row[target_col] = y_pred
        working = pd.concat([working, new_row], ignore_index=True, sort=False)

        last_timestamp = next_time

    return pd.DataFrame(forecast_rows)


def save_forecast_plot(actual_df, forecast_df, date_col, target_col, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_df[date_col], actual_df[target_col], label='Historical', marker='o')
    plt.plot(forecast_df[date_col], forecast_df[target_col], label='Forecast', marker='o')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title('Forecast vs Actual')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def create_interactive_forecast_charts(full_df, last_7_actual, last_7_forecast, next_day_forecast, date_col, target_col):
    """Create interactive charts using plotly for hover functionality."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Chart 1: Last 7 days - Actual vs Forecasted
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=last_7_actual[date_col],
            y=last_7_actual[target_col],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        fig1.add_trace(go.Scatter(
            x=last_7_forecast[date_col],
            y=last_7_forecast[target_col],
            mode='lines+markers',
            name='Forecasted',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        fig1.update_layout(
            title=f'Last 7 Days: Actual vs Forecasted {target_col}',
            xaxis_title='Date',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True
        )

        # Chart 2: Next Day Forecast
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=full_df[date_col].tail(30),  # Show last 30 days for context
            y=full_df[target_col].tail(30),
            mode='lines+markers',
            name='Historical',
            line=dict(color='gray', width=1),
            marker=dict(size=4, opacity=0.7)
        ))
        fig2.add_trace(go.Scatter(
            x=next_day_forecast[date_col],
            y=next_day_forecast[target_col],
            mode='lines+markers',
            name='Next Day Forecast',
            line=dict(color='green', width=3),
            marker=dict(size=8, color='green')
        ))
        fig2.update_layout(
            title=f'Next Day Forecast: {target_col}',
            xaxis_title='Date',
            yaxis_title=target_col,
            hovermode='x unified',
            showlegend=True
        )

        return fig1, fig2
    except ImportError:
        # Fallback if plotly is not available
        return None, None


def _sanitize_feature_names(df, ignore_columns=None):
    """Replace invalid LightGBM feature name chars with underscores.

    ignore_columns: list of columns to leave untouched (e.g., date/target column names).
    """
    df = df.copy()
    ignore_columns = set(ignore_columns or [])

    clean_names = {}
    for col in df.columns:
        if col in ignore_columns:
            clean_names[col] = col
            continue

        # LightGBM does not accept special JSON characters in feature names.
        clean = re.sub(r"[^A-Za-z0-9_]", "_", str(col))
        # Avoid leading digits (optional, but safe for other frameworks)
        if re.match(r"^[0-9]", clean):
            clean = f"f_{clean}"
        clean_names[col] = clean

    return df.rename(columns=clean_names)


def train_forecast_on_full_data(full_df, date_col, target_col, frequency, algorithm):
    """Train forecasting model on full dataset and evaluate on last 7 days."""

    print(f"Training {algorithm} on full dataset with {len(full_df)} samples")

    # Initialize default results with all required keys
    default_result = {
        "algorithm": algorithm,
        "success": False,
        "eval_mape": 0,
        "eval_rmse": 0,
        "eval_r2": 0,
        "error": None
    }

    try:
        # Create time series features for the full dataset
        df_features = create_time_series_features(full_df, target_col, date_col, frequency)
        print(f"Created features for {len(df_features)} samples after feature engineering")

        # Sanitize feature names for LightGBM (no special JSON characters)
        df_features = _sanitize_feature_names(df_features, ignore_columns=[date_col, target_col])

        # Drop any remaining non-numeric columns that LightGBM cannot handle
        non_numeric_cols = df_features.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns.tolist()
        if non_numeric_cols:
            print(f"Dropping non-numeric columns before training: {non_numeric_cols}")
            df_features = df_features.drop(columns=non_numeric_cols)

        # Split for evaluation: use all data except last 7 days for training
        eval_days = get_eval_days_for_frequency(frequency, 7)  # Get last 7 periods
        print(f"Using {eval_days} periods for evaluation")

        if len(df_features) <= eval_days:
            raise ValueError(f"Not enough data: {len(df_features)} samples, need more than {eval_days} for evaluation")

        train_data = df_features.iloc[:-eval_days]
        eval_data = df_features.tail(eval_days)

        print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples")

        if len(train_data) == 0:
            raise ValueError("Not enough data for training after reserving evaluation period")

        # Train model on full training data
        X_train = train_data.drop(columns=[target_col, date_col])
        y_train = train_data[target_col]

        model, param_grid = get_base_model(algorithm)

        if model is None:
            result = default_result.copy()
            result['error'] = "Model not supported"
            result['success'] = False
            return result, None

        # Train with hyperparameter tuning
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) - 1))
        search = RandomizedSearchCV(
            model, param_grid, cv=tscv, scoring='neg_mean_absolute_percentage_error',
            n_iter=10, n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Evaluate on last 7 days
        X_eval = eval_data.drop(columns=[target_col, date_col])
        y_eval_actual = eval_data[target_col]

        y_eval_pred = best_model.predict(X_eval)

        # Create evaluation dataframe
        eval_results = eval_data[[date_col, target_col]].copy()
        eval_results['forecasted'] = y_eval_pred

        # Generate next day forecast
        next_day_forecast = generate_forecast(
            full_df, best_model, date_col, target_col, frequency, steps=1
        )

        # Calculate metrics
        mape = mean_absolute_percentage_error(y_eval_actual, y_eval_pred)
        rmse = np.sqrt(mean_squared_error(y_eval_actual, y_eval_pred))
        r2 = r2_score(y_eval_actual, y_eval_pred)

        results = {
            "algorithm": algorithm,
            "best_params": search.best_params_,
            "success": True,
            "eval_mape": round(mape, 4),
            "eval_rmse": round(rmse, 4),
            "eval_r2": round(r2, 4),
            "eval_period_days": eval_days,
            "training_samples": len(X_train),
            "eval_samples": len(X_eval)
        }

        return results, {
            'model': best_model,
            'eval_results': eval_results,
            'next_day_forecast': next_day_forecast,
            'full_df': full_df
        }
    
    except Exception as e:
        print(f"Error training {algorithm}: {str(e)}")
        result = default_result.copy()
        result['error'] = str(e)
        result['success'] = False
        return result, None


def get_eval_days_for_frequency(frequency, target_days=7):
    """Convert target days to appropriate periods based on frequency."""
    if frequency == '15min':
        return target_days * 24 * 4  # 7 days * 24 hours * 4 quarters
    elif frequency == '30min':
        return target_days * 24 * 2  # 7 days * 24 hours * 2 half-hours
    elif frequency == 'hourly':
        return target_days * 24  # 7 days * 24 hours
    else:  # daily or other
        return target_days

def get_base_model(algorithm):
    if algorithm == "XGBoost":
        return XGBRegressor(objective='reg:squarederror'), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    elif algorithm == "LightGBM":
        return lgb.LGBMRegressor(), {
            'n_estimators': [50, 100, 200],
            'max_depth': [-1, 10, 20],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50]
        }
    
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
