# trainers.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import re
from ml_pipeline import build_pipeline

def get_model_and_params(algo_name, model_type):
    if model_type == "Binary Classification":
        if algo_name == "Logistic Regression":
            return LogisticRegression(), {
                "C": [0.001, 0.01, 0.1, 110, 100],
                "penalty": ["l1"],
                "solver": ["liblinear", "saga"],
                
                "max_iter": [10000]
            }
        elif algo_name == "Random Forest":
            return RandomForestClassifier(), {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None]
            }
        elif algo_name == "SVM":
            return SVC(probability=True), {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto", 0.1, 0.01],
                "class_weight": ["balanced", None]
            }
        elif algo_name == "KNN":
            return KNeighborsClassifier(), {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2]
            }
        elif algo_name == "XGBoost":
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2]
            }

    elif model_type == "Multi-Class Classification":
        if algo_name == "Decision Tree":
            return DecisionTreeClassifier(), {
                "max_depth": [None, 5, 10, 15, 20, 25],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy", "log_loss"],
                "class_weight": ["balanced", None]
            }
        elif algo_name == "Naive Bayes":
            return GaussianNB(), {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            }
        elif algo_name == "SVM":
            return SVC(probability=True), {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.1, 0.01],
                "decision_function_shape": ["ovo", "ovr"],
                "class_weight": ["balanced", None]
            }
        elif algo_name == "Random Forest":
            return RandomForestClassifier(), {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None]
            }
        elif algo_name == "XGBoost":
            return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2]
            }

    elif model_type == "Regression":
        if algo_name == "Linear Regression":
            return Ridge(), {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            }
        elif algo_name == "Ridge":
            return Ridge(), {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
            }
        elif algo_name == "Lasso":
            return Lasso(), {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "selection": ["cyclic", "random"],
                "max_iter": [1000, 3000, 5000]
            }
        elif algo_name == "SVR":
            return SVR(), {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto", 0.1, 0.01],
                "epsilon": [0.1, 0.2, 0.5]
            }
        elif algo_name == "Random Forest Regressor":
            return RandomForestRegressor(), {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif algo_name == "XGBoost":
            return XGBRegressor(), {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2]
            }

    if algo_name == "KMeans":
        return KMeans(), {
            "n_clusters": [2, 3, 4, 5, 7, 10],
            "init": ["k-means++", "random"],
            "algorithm": ["auto", "full", "elkan"]
        }
    elif algo_name == "DBSCAN":
        return DBSCAN(), {
            "eps": [0.1, 0.3, 0.5, 0.7, 1.0],
            "min_samples": [3, 5, 10, 15],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
        }
    elif algo_name == "Agglomerative Clustering":
        return AgglomerativeClustering(), {
            "n_clusters": [2, 3, 5, 7, 10],
            "affinity": ["euclidean", "l1", "l2", "manhattan"],
            "linkage": ["ward", "complete", "average", "single"]
        }
    elif algo_name == "PCA":
        return PCA(), {
            "n_components": [2, 3, 5, 7, 10, 15],
            "svd_solver": ["auto", "full", "arpack", "randomized"]
        }
    elif algo_name == "Isolation Forest":
        return IsolationForest(), {
            "n_estimators": [50, 100, 200],
            "contamination": [0.01, 0.05, 0.1, "auto"],
            "max_samples": [100, "auto"],
            "max_features": [0.5, 0.8, 1.0]
        }
    elif algo_name == "One-Class SVM":
        return OneClassSVM(), {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "nu": [0.01, 0.05, 0.1, 0.5],
            "gamma": ["scale", "auto", 0.1, 0.01]
        }
    elif algo_name == "Autoencoder":
        return None, {}  # Placeholder

    return None, {}

def prefix_param_grid(param_grid, prefix):
    return {f"{prefix}__{k}": v for k, v in param_grid.items()}

def train_with_grid_search(X_train, y_train, model_type, selected_algorithms, X_test=None, y_test=None, selected_columns=None, apply_pca=False, remove_corr=False):
    """
    Trains multiple models using grid search CV based on selected algorithms
    Now uses a pipeline for preprocessing + model.
    """
    results = []
    trained_models = {}

    # Clean y_train if it exists
    if y_train is not None:
        try:
            y_train = pd.to_numeric(y_train, errors='coerce')
            if y_train.isna().any():
                print(f"Warning: {y_train.isna().sum()} NaN values found after numeric conversion")
                y_train = y_train.dropna().reset_index(drop=True)
            print(f"Processed y_train shape: {y_train.shape}")
            print(f"Processed y_train sample: {y_train.head()}")
        except Exception as e:
            print(f"Error cleaning y_train: {e}")

    for algo_name in selected_algorithms:
        print(f"Training {algo_name} for {model_type}")
        model, param_grid = get_model_and_params(algo_name, model_type)
        pipeline = build_pipeline(selected_columns, model, apply_pca=apply_pca, remove_corr=remove_corr, y=y_train, sample_df=X_train)
        if param_grid:
            param_grid = prefix_param_grid(param_grid, 'model')
        if model is None:
            results.append({
                "algorithm": algo_name,
                "error": "Model implementation not available",
                "success": False
            })
            continue
        dataset_size = len(X_train) if X_train is not None else 0
        print("Dataset size:", dataset_size)
        try:
            best_model, best_params = train_single_model(pipeline, param_grid, X_train, y_train, dataset_size=dataset_size)
            print("Best Model:", best_model)
            trained_models[algo_name] = best_model
            result = {
                "algorithm": algo_name,
                "best_params": best_params,
                "success": True
            }
            if y_test is not None:
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
                y_pred = best_model.predict(X_test)
                y_pred_train = best_model.predict(X_train)
                if model_type in ["Binary Classification", "Multi-Class Classification"]:
                    result["Training_accuracy"] = round(accuracy_score(y_train, y_pred_train), 4)
                    result["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
                    result["f1"] = round(f1_score(y_test, y_pred, average='weighted'), 4)
                    result["precision"] = round(precision_score(y_test, y_pred, average='weighted'), 4)
                    result["recall"] = round(recall_score(y_test, y_pred, average='weighted'), 4)
                elif model_type == "Regression":
                    result["mse"] = round(mean_squared_error(y_test, y_pred), 4)
                    result["r2"] = round(r2_score(y_test, y_pred), 4)
            results.append(result)
        except Exception as e:
            print(f"Error training {algo_name}: {str(e)}")
            results.append({
                "algorithm": algo_name,
                "error": str(e),
                "success": False
            })
    return results, trained_models

def train_single_model(model, param_grid, X_train, y_train, cv=2, dataset_size=None):
    """
    Trains a single model using grid search CV
    
    Args:
        model: The sklearn estimator to train
        param_grid: Parameter grid for grid search
        X_train: Feature data for training
        y_train: Target data for training (can be None for unsupervised learning)
        cv: Number of cross-validation folds
        dataset_size: Size of dataset to adjust CV if needed
        
    Returns:
        Tuple of (best_model, best_params)
    """
    if not param_grid:
        model.fit(X_train, y_train)
        return model, None

    # if dataset_size:
    #     if dataset_size < 100 or dataset_size > 10000:
    #         cv = 3

    if y_train is None:
        scoring = None
    elif hasattr(y_train, 'dtype') and y_train.dtype.kind == 'f':
        scoring = 'neg_mean_squared_error'
    elif len(set(y_train)) > 2:
        scoring = 'balanced_accuracy'
    else:
        scoring = 'f1'

    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    print("I am here in grid")
    print(grid.best_estimator_,grid.best_params_)
    return grid.best_estimator_, grid.best_params_
