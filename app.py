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
from forecasting import train_time_series_model, create_time_series_features

# Set matplotlib backend to non-GUI for web server environment
import matplotlib
matplotlib.use('Agg')

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
FORECAST_CHARTS = {}  # Store Plotly charts separately to avoid session cookie overflow
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
 
        # Save original data before feature engineering for forecasting
        original_path = os.path.join(app.config['TEMP_FOLDER'], 'df_original.csv')
        df.to_csv(original_path, index=False)
        session['original_path'] = original_path
 
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
        print("POST request received for select_columns")
        selected_columns = request.form.getlist('columns')
        print(f"Selected columns: {selected_columns}")
        
        # Validate that at least one column is selected
        if not selected_columns:
            flash("Please select at least one column to proceed.", "error")
            return render_template('select_columns.html', columns=columns)
        
        has_target = request.form.get('has_target')
        target_column = request.form.get('target_column') if has_target == 'Yes' else None
        print(f"Has target: {has_target}, Target column: {target_column}")
        
        # Validate target column selection
        if has_target == 'Yes' and not target_column:
            flash("Please select a target column or choose 'No' for unsupervised learning.", "error")
            return render_template('select_columns.html', columns=columns)
 
        session['selected_columns'] = selected_columns
        print("see me selected columns" ,session['selected_columns'])

        date_column = request.form.get('date_column')
        if date_column:
            # Validate date column can be parsed as datetime
            try:
                pd.to_datetime(df_engineered[date_column], errors='raise')
            except Exception as e:
                flash(f"Selected date column '{date_column}' cannot be parsed as datetime: {e}", "error")
                return render_template('select_columns.html', columns=columns)
            
            # Ensure date column is included in selected columns
            if date_column not in selected_columns:
                selected_columns.append(date_column)
            
            session['date_column'] = date_column  # Save date column
 
        # Save the target column in session (if any)
        session['target_column'] = target_column
 
        # Keep has_target true only when an actual target is selected
        session['has_target'] = bool(target_column)
 
        # Forecasting requires both a date column and a target column.
        session['can_forecast'] = bool(target_column and date_column)
 
        if target_column:
            X_train, X_test, y_train, y_test = preprocess_data(df_engineered, selected_columns, target_column)
 
            # Save target variables correctly - no header row, just the values
            pd.DataFrame(y_train).to_csv(os.path.join(app.config['TEMP_FOLDER'], 'y_train.csv'), index=False, header=False)
            pd.DataFrame(y_test).to_csv(os.path.join(app.config['TEMP_FOLDER'], 'y_test.csv'), index=False, header=False)
 
            session['y_train_shape'] = len(y_train)
            session['y_test_shape'] = len(y_test)
        else:
            X_train, X_test, _, _ = preprocess_data(df_engineered, selected_columns)
            session.pop('y_train_shape', None)
            session.pop('y_test_shape', None)
 
        X_train.to_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_test.csv'), index=False)
        session['preprocessing_done'] = True
        session['X_train_shape'] = X_train.shape
        session['X_test_shape'] = X_test.shape
 
        print(f"Preprocessing completed. Redirecting to preprocessing_result. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
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
            'regression': 'Regression',
            'forecasting': 'Time Series Forecasting'
        }
 
        model_type = model_type_mapping.get(form_model_type)
        if not model_type:
            return "Error: Invalid model type selected.", 400
 
        session['model_type'] = model_type
        
        # New: Handle Forecasting specific inputs
        if form_model_type == 'forecasting':
            session['forecasting_domain'] = request.form.get('forecasting_domain')
            session['frequency'] = request.form.get('frequency')
            # Ensure required forecasting columns are selected
            if not session.get('can_forecast'):
                return ("Error: Time series forecasting requires both a date column & a target column. "
                        "Please go back and select both."), 400

        return redirect(url_for('select_algorithms', model_type=form_model_type))
 
    return render_template(
        'model_selection.html',
        has_target=session.get('has_target'),
        can_forecast=session.get('can_forecast', False)
    )
 
@app.route('/select-algorithms/<model_type>', methods=['GET', 'POST'])
def select_algorithms(model_type):
    model_type_mapping = {
        'binary_classification': 'Binary Classification',
        'multi_class_classification': 'Multi-Class Classification',
        'regression': 'Regression',
        'forecasting': 'Time Series Forecasting'
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
        "Binary Classification": ["Isolation Forest", "One-Class SVM", "KMeans", "Agglomerative Clustering"],
        "Multi-Class Classification": ["Isolation Forest", "One-Class SVM", "KMeans", "Agglomerative Clustering"],
        "Regression": ["Isolation Forest", "One-Class SVM", "KMeans", "Agglomerative Clustering"]
    }
    
    forecasting_algos = ["XGBoost", "LightGBM"]

    if display_model_type == 'Time Series Forecasting':
        algorithms = forecasting_algos
    else:
        algorithms = supervised[display_model_type] if has_target else unsupervised[display_model_type]
 
    if request.method == 'POST':
        selected_algorithms = request.form.getlist('selected_algorithms')
        if not selected_algorithms:
            return "Error: No algorithms selected.", 400
 
        session['selected_algorithms'] = selected_algorithms
        # Save hyperparameter tuning preference (defaults to False)
        enable_tuning = request.form.get('enable_tuning', 'no') == 'yes'
        session['enable_tuning'] = enable_tuning
        return redirect(url_for('train_selected_models'))
 
    return render_template('algorithm_selection.html', model_type=display_model_type, algorithms=algorithms, has_target=has_target)
 
@app.route('/train-models', methods=['GET', 'POST'])
def train_selected_models():
    try:
        model_type = session.get('model_type')
        
        # Handle Time Series Forecasting separately - it doesn't use X_train/X_test files
        if model_type == 'Time Series Forecasting':
            return train_forecasting_models()
        
        # Regular ML training - load preprocessed data from temp folder
        try:
            X_train = pd.read_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_train.csv'))
            X_test = pd.read_csv(os.path.join(app.config['TEMP_FOLDER'], 'X_test.csv'))
        except Exception as e:
            return f"Error loading preprocessed data: {str(e)}", 500
        
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
        enable_tuning = session.get('enable_tuning', False)

        if not selected_algorithms or len(selected_algorithms) == 0:
            return "Error: No algorithms selected."
 
        results, trained_models = train_with_grid_search(
            X_train, y_train, model_type, selected_algorithms, X_test, y_test,
            selected_columns=selected_columns, apply_pca=apply_pca, remove_corr=remove_corr,
            enable_tuning=enable_tuning
        )
 
        # ✅ Store training results in session
        session['training_results'] = results
 
        # ✅ Store trained models globally
        global TRAINED_MODELS
        TRAINED_MODELS.clear()
        TRAINED_MODELS.update(trained_models)
 
        return redirect(url_for('training_results'))
 
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Unexpected error during model training: {str(e)}", 500


def train_forecasting_models():
    """Handle time series forecasting training separately."""
    try:
        date_col = session.get('date_column')
        frequency = session.get('frequency')
        selected_algorithms = session.get('selected_algorithms')
        
        if not selected_algorithms or len(selected_algorithms) == 0:
            return "Error: No algorithms selected for forecasting.", 400
        
        # Load the original dataset (before feature engineering) for forecasting
        original_path = session.get('original_path')
        if not original_path:
            # Fallback to original uploaded file if original_path not set
            original_path = session.get('filepath')
        
        if not original_path:
            return "Error: No data file found. Please upload a dataset first.", 400
        
        try:
            original_df = pd.read_csv(original_path)
        except Exception as e:
            return f"Error: Could not load data file: {str(e)}", 400
        
        # Ensure the selected timestamp and target columns exist
        if not date_col or date_col not in original_df.columns:
            return ("Error: Date column missing from original data. "
                    "Please ensure you select a date column in the previous step."), 400
        
        target_col = session.get('target_column')
        if not target_col or target_col not in original_df.columns:
            return ("Error: Target column missing from original data."), 400

        # Only use the date and target columns for forecasting
        original_df = original_df[[date_col, target_col]]
        
        try:
            # Validate that date column can be parsed as datetime.
            original_df[date_col] = pd.to_datetime(original_df[date_col], errors='raise')
        except Exception as e:
            return (f"Error: Selected date column '{date_col}' is not valid datetime data: {e}. "
                    "Please choose a proper timestamp column."), 400

        try:
            algo_results = []
            algo_models = {}
            
            for algo in selected_algorithms:
                print(f"Training {algo} for forecasting...")
                from forecasting import train_forecast_on_full_data, create_interactive_forecast_charts
                
                results, model_data = train_forecast_on_full_data(
                    original_df, date_col, target_col, frequency, algo
                )
                
                print(f"Training completed for {algo}: success={results.get('success', False)}")
                
                if results['success'] and model_data:
                    print(f"Creating charts for {algo}...")
                    # Create interactive charts
                    eval_results = model_data['eval_results']
                    next_day_forecast = model_data['next_day_forecast']
                    full_df = model_data['full_df']
                    
                    # Get last 7 days data for evaluation chart
                    last_7_actual = eval_results[[date_col, target_col]]
                    last_7_forecast = eval_results[[date_col, 'forecasted']].rename(columns={'forecasted': target_col})
                    
                    # Create interactive charts
                    fig_eval, fig_forecast = create_interactive_forecast_charts(
                        full_df, last_7_actual, last_7_forecast, next_day_forecast, date_col, target_col
                    )
                    
                    if fig_eval and fig_forecast:
                        # Store charts in global dict instead of session to avoid cookie overflow
                        FORECAST_CHARTS[algo] = {
                            'eval_chart': fig_eval.to_html(full_html=False, include_plotlyjs='cdn'),
                            'forecast_chart': fig_forecast.to_html(full_html=False, include_plotlyjs='cdn')
                        }
                        results['has_charts'] = True
                        print(f"Charts created successfully for {algo}")
                    else:
                        results['has_charts'] = False
                        print(f"Chart creation failed for {algo}")
                else:
                    results['has_charts'] = False
                
                algo_results.append(results)
                if model_data and 'model' in model_data:
                    algo_models[algo] = model_data['model']
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in Time Series Processing: {str(e)}\nMake sure you selected the Date Column in the previous step.", 500

        # Store results in session (only JSON-serializable data)
        session['training_results'] = algo_results
        
        # Store trained models in global dictionary (NOT in session - models are not JSON serializable)
        global TRAINED_MODELS
        TRAINED_MODELS.clear()
        TRAINED_MODELS.update(algo_models)
        
        return redirect(url_for('training_results'))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Unexpected error during forecasting training: {str(e)}", 500


@app.route('/training-results')
def training_results():
    results = session.get('training_results', [])
    model_type = session.get('model_type', '')
    return render_template('training_result.html', results=results, model_type=model_type, forecast_charts=FORECAST_CHARTS)

@app.route('/get-forecast-chart/<algorithm>/<chart_type>')
def get_forecast_chart(algorithm, chart_type):
    """Retrieve forecast chart HTML from in-memory storage."""
    if algorithm in FORECAST_CHARTS:
        if chart_type == 'eval':
            return FORECAST_CHARTS[algorithm].get('eval_chart', '')
        elif chart_type == 'forecast':
            return FORECAST_CHARTS[algorithm].get('forecast_chart', '')
    return ''
 
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

        # Unsupervised Labeling: Map -1 to "Anomaly" for easy reading
        try:
            if 'model' in pipeline.named_steps:
                model_step = pipeline.named_steps['model']
                model_name = type(model_step).__name__
                if model_name in ['IsolationForest', 'OneClassSVM']:
                    predictions = ["Anomaly" if p == -1 else "Normal" for p in predictions]
        except Exception as e:
            print(f"Could not apply unsupervised labeling: {e}")

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
 
 
 