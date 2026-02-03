import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

def perform_feature_engineering(df, actions, remove_correlated=False, apply_pca=False, outlier_actions=None):
    df = df.copy()

    for col, action in actions.items():
        if action == 'drop_column':
            df.drop(columns=[col], inplace=True)
        elif action == 'fill_mean':
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
        elif action == 'fill_median':
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        elif action == 'fill_mode':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif action == 'encode':
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    # Handle outliers
    if outlier_actions:
        for col, outlier_action in outlier_actions.items():
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                if outlier_action == 'remove_row':
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                elif outlier_action == 'fill_mean':
                    mean_val = df[(df[col] >= lower) & (df[col] <= upper)][col].mean()
                    df[col] = np.where((df[col] < lower) | (df[col] > upper), mean_val, df[col])
                elif outlier_action == 'fill_median':
                    median_val = df[(df[col] >= lower) & (df[col] <= upper)][col].median()
                    df[col] = np.where((df[col] < lower) | (df[col] > upper), median_val, df[col])
                elif outlier_action == 'fill_mode':
                    mode_val = df[(df[col] >= lower) & (df[col] <= upper)][col].mode()[0]
                    df[col] = np.where((df[col] < lower) | (df[col] > upper), mode_val, df[col])
                # Do nothing if action is 'do_nothing'

    if remove_correlated:
        corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        df.drop(columns=to_drop, inplace=True)

    if apply_pca:
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(numeric_df.fillna(0))
            df['PCA1'] = pca_components[:, 0]
            df['PCA2'] = pca_components[:, 1]
            df.drop(columns=numeric_df.columns, inplace=True)

    return df

def generate_feature_engineering_graphs(df, static_dir='static'):
    os.makedirs(static_dir, exist_ok=True)
    plot_paths = []
    # 1. Feature distributions
    for col in df.columns:
        plt.figure(figsize=(4,3))
        if df[col].dtype in ['float64', 'int64']:
            sns.histplot(df[col].dropna(), kde=True, bins=20)
            plt.title(f'Distribution: {col}')
        else:
            vc = df[col].value_counts().head(20)  # limit to top 20 for clarity
            sns.barplot(x=vc.values, y=vc.index)
            plt.title(f'Value Counts: {col}')
        plt.tight_layout()
        path = os.path.join(static_dir, f'feature_{col}.png')
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)
    # 2. Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_path = None
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(min(12, 1.2*numeric_df.shape[1]), 8))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        corr_path = os.path.join(static_dir, 'correlation_heatmap.png')
        plt.savefig(corr_path)
        plt.close()
    return plot_paths, corr_path
