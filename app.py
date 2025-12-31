from flask import Flask, render_template, request, redirect, url_for, session,flash, send_from_directory
import joblib
import pandas as pd
import os
import psycopg2
from werkzeug.utils import secure_filename
from preprocessing import preprocess_data
from feature_engineering import perform_feature_engineering, detect_outliers_iqr, generate_feature_engineering_graphs
from trainers import train_with_grid_search
import re
import pickle
import traceback
 
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MODEL_FOLDER']= 'my_models'
 
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
 
if not os.path.exists(app.config['TEMP_FOLDER']):
    os.makedirs(app.config['TEMP_FOLDER'])
   
if not os.path.exists(app.config['MODEL_FOLDER']) :
     os.makedirs(app.config['MODEL_FOLDER'])  
 
if not os.path.exists('Predicted Datasets'):
    os.makedirs('Predicted Datasets')
 
TRAINED_MODELS = {}
import psycopg2
 
def get_postgres_connection():
    return psycopg2.connect("dbname='smartpulse' user='postgres' host='13.127.94.9' password='cms123' port ='5433'"
    )
   
def get_table_names():
    conn = get_postgres_connection()
   
    cur = conn.cursor()
    cur.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';""")
 
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return tables
 
def load_table_as_df(table_name):
    conn = get_postgres_connection()
    query = f'SELECT * FROM "{table_name}"'
    df = pd.read_sql(query, conn)
    conn.close()
    return df
 
 
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             session['filepath'] = filepath
#             return redirect(url_for('feature_engineering'))
#     return render_template('upload.html')
 
 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        selected_table = request.form.get('table_name')
        if selected_table:
            df = load_table_as_df(selected_table)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'from_postgres.csv')
            df.to_csv(filepath, index=False)
            session['filepath'] = filepath
            return redirect(url_for('feature_engineering'))
 
    table_names = get_table_names()
    saved_models = []
    for f in os.listdir(app.config['MODEL_FOLDER']):
        if f.endswith('.pkl'):
            saved_models.append(f[:-4])  # remove .pkl extension
 
    return render_template('upload.html', saved_models=saved_models, table_names=table_names)
 
   
 
@app.route('/delete_model/<model_name>', methods=['POST'])
def delete_model(model_name):
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name + '.pkl')
        if os.path.exists(model_path):
            os.remove(model_path)
            flash(f'Model "{model_name}" deleted successfully!', 'success')
        else:
            flash(f'Model "{model_name}" not found.', 'error')
    except Exception as e:
        flash(f'Error deleting model: {str(e)}', 'error')
        traceback.print_exc()
    return redirect(url_for('upload_file'))
 
 
@app.route('/feature-engineering', methods=['GET', 'POST'])
def feature_engineering():
    filepath = session.get('filepath')
    if not filepath:
        return redirect(url_for('upload_file'))
 
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
    session['original_columns'] = df.columns.tolist()
 
    if request.method == 'POST':
        actions = {col: request.form.get(col) for col in df.columns}
        outlier_actions = {
            col: request.form.get(f"outlier_{col}")
            for col in df.select_dtypes(include=['float64', 'int64']).columns
        }
        remove_corr = 'remove_correlated_features' in request.form
        apply_pca = 'apply_pca' in request.form

        session['apply_pca'] = apply_pca
        session['remove_correlated_features'] = remove_corr

        df_engineered = perform_feature_engineering(df, actions, remove_corr, apply_pca, outlier_actions)
 
        engineered_path = os.path.join(app.config['TEMP_FOLDER'], 'df_engineered.csv')
        df_engineered.to_csv(engineered_path, index=False)
        session['engineered_path'] = engineered_path
        return redirect(url_for('select_columns'))
 
    column_info = []
    for col in df.columns:
        missing_pct = round(df[col].isnull().mean() * 100, 2)
        dtype = str(df[col].dtype)
        column_info.append((col, dtype, missing_pct))
 
    # Get outlier information using the centralized function
    outlier_info = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        outliers = detect_outliers_iqr(df[col])
        outlier_pct = round((outliers.sum() / len(df)) * 100, 2)
        dtype = str(df[col].dtype)
        outlier_info.append((col, dtype, outlier_pct))
 
    # Generate feature engineering graphs
    feature_plots, corr_plot = generate_feature_engineering_graphs(df)
    feature_plots = [os.path.relpath(p, start='static') for p in feature_plots]
    corr_plot = os.path.relpath(corr_plot, start='static') if corr_plot else None

    return render_template(
        'feature_engineering_config.html',
        column_info=column_info,
        outlier_info=outlier_info,
        shape=df.shape,
        feature_plots=feature_plots,
        corr_plot=corr_plot
    )
 
@app.route('/select-columns', methods=['GET', 'POST'])
def select_columns():
    if 'engineered_path' not in session:
        return redirect(url_for('feature_engineering'))
 
    df_engineered = pd.read_csv(session.get('engineered_path'))
    columns = list(zip(df_engineered.columns.tolist(), df_engineered.dtypes.astype(str).tolist()))
 
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')
        has_target = request.form.get('has_target')
        target_column = request.form.get('target_column') if has_target == 'Yes' else None
 
        session['selected_columns'] = selected_columns
        print("see me selected columns" ,session['selected_columns'])
 
        if target_column:
            X_train, X_test, y_train, y_test = preprocess_data(df_engineered, selected_columns, target_column)
 
            # Save target variables correctly - no header row, just the values
            pd.DataFrame(y_train).to_csv(os.path.join(app.config['TEMP_FOLDER'], 'y_train.csv'), index=False, header=False)
            pd.DataFrame(y_test).to_csv(os.path.join(app.config['TEMP_FOLDER'], 'y_test.csv'), index=False, header=False)
 
            session['has_target'] = True
            session['y_train_shape'] = len(y_train)
            session['y_test_shape'] = len(y_test)
        else:
            X_train, X_test, _, _ = preprocess_data(df_engineered, selected_columns)
            session['has_target'] = False
 
        X_train.to_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_test.csv'), index=False)
        session['preprocessing_done'] = True
        session['X_train_shape'] = X_train.shape
        session['X_test_shape'] = X_test.shape
 
        return redirect(url_for('preprocessing_result'))
 
    return render_template('select_columns.html', columns=columns)
 
@app.route('/preprocessing-result')
def preprocessing_result():
    if not session.get('preprocessing_done'):
        return redirect(url_for('select_columns'))
 
    return render_template('preprocessing_result.html',
                           X_train_shape=session.get('X_train_shape'),
                           X_test_shape=session.get('X_test_shape'),
                           y_train_shape=session.get('y_train_shape'),
                           y_test_shape=session.get('y_test_shape'),
                           target_selected=session.get('has_target'))
 
@app.route('/model-training', methods=['GET', 'POST'])
def model_training():
    if not session.get('preprocessing_done'):
        return redirect(url_for('select_columns'))
 
    if request.method == 'POST':
        form_model_type = request.form.get('model_type')
        if not form_model_type:
            return "Error: No model type selected.", 400
 
        model_type_mapping = {
            'binary_classification': 'Binary Classification',
            'multi_class_classification': 'Multi-Class Classification',
            'regression': 'Regression'
        }
 
        model_type = model_type_mapping.get(form_model_type)
        if not model_type:
            return "Error: Invalid model type selected.", 400
 
        session['model_type'] = model_type
        return redirect(url_for('select_algorithms', model_type=form_model_type))
 
    return render_template('model_selection.html')
 
@app.route('/select-algorithms/<model_type>', methods=['GET', 'POST'])
def select_algorithms(model_type):
    model_type_mapping = {
        'binary_classification': 'Binary Classification',
        'multi_class_classification': 'Multi-Class Classification',
        'regression': 'Regression'
    }
 
    display_model_type = model_type_mapping.get(model_type)
    if not display_model_type:
        return "Error: Invalid model type route.", 400
 
    session['model_type'] = display_model_type
    has_target = session.get('has_target', False)
 
    supervised = {
        "Binary Classification": ["Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost"],
        "Multi-Class Classification": ["Decision Tree", "Naive Bayes", "SVM", "Random Forest", "XGBoost"],
        "Regression": ["Linear Regression", "Ridge", "Lasso", "SVR", "Random Forest Regressor"]
    }
 
    unsupervised = {
        "Binary Classification": ["KMeans", "DBSCAN", "Isolation Forest"],
        "Multi-Class Classification": ["KMeans", "Agglomerative Clustering"],
       
}
 
    algorithms = supervised[display_model_type] if has_target else unsupervised[display_model_type]
 
    if request.method == 'POST':
        selected_algorithms = request.form.getlist('selected_algorithms')
        if not selected_algorithms:
            return "Error: No algorithms selected.", 400
 
        session['selected_algorithms'] = selected_algorithms
        return redirect(url_for('train_selected_models'))
 
    return render_template('algorithm_selection.html', model_type=display_model_type, algorithms=algorithms)
 
@app.route('/train-models', methods=['GET', 'POST'])
def train_selected_models():
    try:
        X_train = pd.read_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_test.csv'))
 
        if X_train.empty or X_test.empty:
            return "Error: X_train or X_test is empty."
 
        if session.get('has_target'):
            try:
                y_train = pd.read_csv(os.path.join(app.config['TEMP_FOLDER'], 'y_train.csv'), header=None).squeeze()
                y_test = pd.read_csv(os.path.join(app.config['TEMP_FOLDER'], 'y_test.csv'), header=None).squeeze()
 
                y_train = pd.to_numeric(y_train, errors='coerce')
                y_test = pd.to_numeric(y_test, errors='coerce')
 
                if y_train.isna().any():
                    y_train = y_train.dropna().reset_index(drop=True)
                if y_test.isna().any():
                    y_test = y_test.dropna().reset_index(drop=True)
 
            except Exception as e:
                return f"Error loading target variables: {str(e)}", 500
        else:
            y_train = y_test = None
 
        model_type = session.get('model_type')
        selected_algorithms = session.get('selected_algorithms')
        selected_columns = session.get('selected_columns')
        apply_pca = session.get('apply_pca', False)
        remove_corr = session.get('remove_correlated_features', False)

        if not model_type or not selected_algorithms or len(selected_algorithms) == 0:
            return "Error: Model type or selected algorithms are not specified."
 
        # ✅ Get results and models
        results, trained_models = train_with_grid_search(
            X_train, y_train, model_type, selected_algorithms, X_test, y_test,
            selected_columns=selected_columns, apply_pca=apply_pca, remove_corr=remove_corr
        )
 
        # ✅ Store training results in session
        session['training_results'] = results
 
        # ✅ Store trained models globally
        global TRAINED_MODELS
        TRAINED_MODELS.clear()
        TRAINED_MODELS.update(trained_models)
 
        return redirect(url_for('training_results'))
 
    except Exception as e:
        return f"Unexpected error during model training: {str(e)}", 500
        traceback.print_exc()
 
 
@app.route('/training-results')
def training_results():
    results = session.get('training_results', [])
    model_type = session.get('model_type', '')
    return render_template('training_result.html', results=results, model_type=model_type)
 
@app.route('/save_model', methods=['POST'])
def save_model():
    global TRAINED_MODELS  # Access the global dictionary
   
    model_name = request.form.get('model_name')
    algorithm = request.form.get('algorithm')
   
    print(f"Attempting to save model: {algorithm} as {model_name}")
    print(f"Available models: {list(TRAINED_MODELS.keys())}")
   
    if not model_name or not algorithm:
        flash("Error: Missing model name or algorithm.")
        return redirect(url_for('training_results'))
   
    model = TRAINED_MODELS.get(algorithm)
   
    if model:
        try:
            # Create a safe filename from the model name
            safe_name = re.sub(r'[^\w\-_.]', '_', model_name)
            if not safe_name.endswith('.pkl'):
                safe_name += '.pkl'
            save_path = os.path.join(app.config['MODEL_FOLDER'], safe_name)
            # Save the pipeline using joblib
            joblib.dump(model, save_path)
            flash(f"Model '{model_name}' saved successfully to {save_path}")
            print(f"Model saved to {save_path}")
        except Exception as e:
            flash(f"Error saving model: {str(e)}")
            print(f"Error saving model: {str(e)}")
            traceback.print_exc()
    else:
        flash(f"Failed to save model. Model '{algorithm}' not found.")
        print(f"Model '{algorithm}' not found in TRAINED_MODELS dictionary")
   
    # Redirect back to the training results page
    return redirect(url_for('training_results'))
 
@app.route('/predict/<model_name>', methods=['GET', 'POST'])
def predict(model_name):
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash('No file uploaded.', 'error')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

        # Load pipeline
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name + '.pkl')
        if not os.path.exists(model_path):
            flash('Model not found.', 'error')
            return redirect(url_for('upload_file'))
        pipeline = joblib.load(model_path)

        # Get feature names from the pipeline
        try:
            selector = pipeline.named_steps['selector']
            feature_names = selector.columns
        except Exception:
            flash('Could not determine feature names from pipeline.', 'error')
            return redirect(url_for('upload_file'))

        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            flash(f'Missing features in uploaded data: {missing_features}', 'error')
            return redirect(request.url)
        df = df[list(feature_names)]
        df = df.dropna()

        # Predict
        predictions = pipeline.predict(df)

        # Try to inverse transform predictions if label encoder exists
        # Assume the target column name is stored in session or can be inferred
        target_column = session.get('target_column')
        encoder_path = os.path.join(app.config['MODEL_FOLDER'], f'label_encoder_{target_column}.pkl') if target_column else None
        if encoder_path and os.path.exists(encoder_path):
            try:
                 
                label_encoder = joblib.load(encoder_path)
                predictions = label_encoder.inverse_transform(predictions)
            except Exception as e:
                print(f"Warning: Could not inverse transform predictions: {e}")

        df['Predicted'] = predictions

        # Save predicted dataset
        pred_filename = f"predicted_{model_name}_{filename}"
        pred_path = os.path.join('Predicted Datasets', pred_filename)
        df.to_csv(pred_path, index=False)
        flash(f'Prediction complete. Download: {pred_filename}', 'success')
        return render_template('predict_result.html', download_link=pred_filename)

    return render_template('predict_upload.html', model_name=model_name)
 
@app.route('/Predicted Datasets/<path:filename>')
def download_predicted(filename):
    return send_from_directory('Predicted Datasets', filename, as_attachment=True)
 
if __name__ == '__main__':
    app.run(debug=True)
 
def prefix_param_grid(param_grid, prefix):
    return {f"{prefix}__{k}": v for k, v in param_grid.items()}
 
 
 