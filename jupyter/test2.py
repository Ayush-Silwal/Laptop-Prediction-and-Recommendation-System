import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request
import os
from customModel import DecisionTree, RandomForest, CustomKNN

app = Flask(__name__)

# Default dropdown options
DEFAULT_COMPANIES = ['Dell', 'HP', 'Lenovo', 'Asus', 'Apple', 'Acer', 'MSI', 'Toshiba', 'Huawei', 'Microsoft']
DEFAULT_TYPES = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook']
DEFAULT_CPUS = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor']
DEFAULT_GPUS = ['Intel', 'AMD', 'Nvidia']
DEFAULT_OSS = ['Windows', 'Mac', 'Others/No OS/Linux']

# Globals
df = pd.DataFrame()
preprocessor = None
rf_model = None
knn_model = None
companies = DEFAULT_COMPANIES
types = DEFAULT_TYPES
cpus = DEFAULT_CPUS
gpus = DEFAULT_GPUS
oss = DEFAULT_OSS

def load_models():
    global df, preprocessor, rf_model, knn_model, companies, types, cpus, gpus, oss

    try:
        # Use the absolute path provided
        model_path = r'C:\Users\Lenovo\Desktop\project 2.0\jupyter\laptop_models_full_custom.pkl'
        print(f"Current working directory: {os.getcwd()}")
        print(f"Trying to load model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        loaded_data = joblib.load(model_path)
        print(f"Loaded data keys: {loaded_data.keys()}")

        if not isinstance(loaded_data, dict):
            raise ValueError("Invalid model format - expected dictionary")

        df = loaded_data.get('df', pd.DataFrame())
        preprocessor = loaded_data.get('preprocessor')
        rf_model = loaded_data.get('random_forest')
        knn_model = loaded_data.get('knn')

        if df.empty or preprocessor is None or rf_model is None or knn_model is None:
            raise ValueError("Essential model components missing")

        companies = sorted(df['Company'].unique().tolist())
        types = sorted(df['TypeName'].unique().tolist())
        cpus = sorted(df['Cpu brand'].unique().tolist()) if 'Cpu brand' in df.columns else DEFAULT_CPUS
        gpus = sorted(df['Gpu brand'].unique().tolist()) if 'Gpu brand' in df.columns else DEFAULT_GPUS
        oss = sorted(df['os'].unique().tolist()) if 'os' in df.columns else DEFAULT_OSS

        print("Model loaded successfully.")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Using default values.")

# Load models when the server starts
load_models()

def calculate_ppi(resolution, screen_size):
    """Calculate PPI from resolution (e.g., '1920x1080') and screen size (inches)."""
    try:
        width, height = map(int, resolution.split('x'))
        diagonal_pixels = np.sqrt(width**2 + height**2)
        return diagonal_pixels / float(screen_size)
    except Exception as e:
        raise ValueError(f"Invalid resolution or screen size: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html',
                           companies=companies,
                           types=types,
                           cpus=cpus,
                           gpus=gpus,
                           oss=oss,
                           model_loaded=(preprocessor is not None and rf_model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    try:
        if preprocessor is None or rf_model is None or knn_model is None:
            raise Exception("Model not loaded properly. Please check the model files.")

        # Map form fields to model-expected fields
        model_data = {
            'Company': form_data.get('company'),
            'TypeName': form_data.get('type'),
            'Ram': form_data.get('ram'),
            'Weight': form_data.get('weight'),
            'Touchscreen': 1 if form_data.get('touchscreen') == 'Yes' else 0,
            'Ips': 1 if form_data.get('ips') == 'Yes' else 0,
            'Cpu brand': form_data.get('cpu'),
            'Gpu brand': form_data.get('gpu'),
            'HDD': form_data.get('HDD'),
            'SSD': form_data.get('SSD'),
            'os': form_data.get('os')
        }

        # Calculate PPI from resolution and screen_size
        resolution = form_data.get('resolution')
        screen_size = form_data.get('screen_size')
        model_data['ppi'] = calculate_ppi(resolution, screen_size)

        input_df = pd.DataFrame([model_data])

        # Convert numeric fields
        for col in ['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Server-side validation
        if input_df[['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']].isna().any().any():
            raise ValueError("Invalid input: Please ensure all numeric fields are valid numbers.")
        if float(input_df['Weight'].iloc[0]) < 1 or float(input_df['Weight'].iloc[0]) > 4:
            raise ValueError("Weight must be between 1 kg and 4 kg.")
        if float(input_df['ppi'].iloc[0]) < 100 or float(input_df['ppi'].iloc[0]) > 500:
            raise ValueError("Calculated PPI must be between 100 and 500.")
        if int(input_df['HDD'].iloc[0]) == 0 and int(input_df['SSD'].iloc[0]) == 0:
            raise ValueError("You must select a non-zero value for either HDD or SSD.")

        # Preprocess and predict
        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()

        # Price prediction using RandomForest
        prediction = rf_model.predict(X_transformed)
        predicted_price = np.exp(prediction[0])  # Reverse log transformation
        formatted_price = f"{predicted_price:,.2f}"

        # Recommendations using CustomKNN
        recommendations = []
        if knn_model:
            X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()

            # Calculate cosine similarities
            distances = []
            for i, sample in enumerate(X_train_transformed):
                cosine_sim = knn_model._cosine_similarity(X_transformed[0], sample)
                distances.append((cosine_sim, i))

            # Get top 5 similar laptops
            top_indices = [idx for _, idx in sorted(distances, reverse=True)[:5]]
            
            for idx in top_indices:
                rec = df.iloc[idx].to_dict()
                rec['Price'] = f"₹{rec['Price']:,.2f}"
                recommendations.append(rec)

        return render_template('index.html',
                               predicted_price=formatted_price,
                               recommendations=recommendations,
                               companies=companies,
                               types=types,
                               cpus=cpus,
                               gpus=gpus,
                               oss=oss,
                               form_data=form_data,
                               model_loaded=True)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)
        return render_template('index.html',
                               error=error_msg,
                               companies=companies,
                               types=types,
                               cpus=cpus,
                               gpus=gpus,
                               oss=oss,
                               form_data=form_data,
                               model_loaded=(preprocessor is not None and rf_model is not None))

if __name__ == '__main__':
    app.run(debug=True)