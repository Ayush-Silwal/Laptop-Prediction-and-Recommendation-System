import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
from customModel import DecisionTree, RandomForest, CustomKNN, CustomKMeans
from db_connection import create_connection, close_connection

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'  # Change in production

# Default dropdown options
DEFAULT_COMPANIES = ['Dell', 'HP', 'Lenovo', 'Asus', 'Apple', 'Acer', 'MSI', 'Toshiba', 'Huawei', 'Microsoft']
DEFAULT_TYPES = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook']
DEFAULT_CPUS = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor']
DEFAULT_GPUS = ['Intel', 'AMD', 'Nvidia']
DEFAULT_OSS = ['Windows', 'Mac', 'Others/No OS/Linux']

# Default cluster names (fallback for older models)
DEFAULT_CLUSTER_NAMES = {
    0: "Budget-Friendly Laptops",
    1: "Mid-Range Performance",
    2: "Premium Workstations",
    3: "Gaming & High-Performance",
    4: "Ultraportable & Business"
}

# Global variables
df = pd.DataFrame()
preprocessor = None
rf_model = None
knn_model = None
kmeans_model = None
companies = DEFAULT_COMPANIES
types = DEFAULT_TYPES
cpus = DEFAULT_CPUS
gpus = DEFAULT_GPUS
oss = DEFAULT_OSS

def load_models():
    global df, preprocessor, rf_model, knn_model, kmeans_model, companies, types, cpus, gpus, oss
    try:
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
        kmeans_model = loaded_data.get('kmeans')
        metadata = loaded_data.get('metadata', {})

        if df.empty or preprocessor is None or rf_model is None or knn_model is None or kmeans_model is None:
            raise ValueError("Essential model components missing")

        companies = sorted(df['Company'].unique().tolist())
        types = sorted(df['TypeName'].unique().tolist())
        cpus = sorted(df['Cpu brand'].unique().tolist()) if 'Cpu brand' in df.columns else DEFAULT_CPUS
        gpus = sorted(df['Gpu brand'].unique().tolist()) if 'Gpu brand' in df.columns else DEFAULT_GPUS
        oss = sorted(df['os'].unique().tolist()) if 'os' in df.columns else DEFAULT_OSS

        # Load feature weights from saved model or try to load from feature_config.json
        try:
            import json
            with open('feature_config.json', 'r') as f:
                feature_config = json.load(f)
                feature_weights = np.array(feature_config['weights'])
                print(f"Loaded feature weights from config file: {len(feature_weights)} features")
        except:
            # Fallback: use RF feature importance if available
            if hasattr(rf_model, 'feature_importances_') and rf_model.feature_importances_ is not None:
                feature_weights = rf_model.feature_importances_.copy()
                feature_weights = feature_weights / feature_weights.max()  # Normalize
                feature_weights = feature_weights * 2 + 0.5  # Scale to [0.5, 2.5]
                print(f"Using Random Forest feature importance as weights: {len(feature_weights)} features")
            else:
                # Create balanced weights
                sample_transform = preprocessor.transform(df.head(1).drop(columns=['Price']))
                if hasattr(sample_transform, 'toarray'):
                    sample_transform = sample_transform.toarray()
                actual_feature_count = sample_transform.shape[1]
                
                # Enhanced feature weights based on domain knowledge
                base_weights = [
                    2.5,  # RAM - very important for performance
                    2.0,  # Storage (SSD/HDD) - important for speed
                    1.8,  # Weight - portability factor
                    1.8,  # PPI - display quality
                    1.5,  # Price-related features
                    1.2,  # CPU - baseline importance
                    1.2,  # GPU - baseline importance
                    1.0,  # Touchscreen/IPS
                    0.8,  # Secondary features
                ]
                
                # Extend to match actual feature count
                feature_weights = np.ones(actual_feature_count)
                for i, weight in enumerate(base_weights):
                    if i < actual_feature_count:
                        feature_weights[i] = weight
                
                print(f"Using default enhanced feature weights: {actual_feature_count} features")

        # Configure enhanced KNN settings
        if hasattr(knn_model, 'set_feature_weights'):
            knn_model.set_feature_weights(feature_weights)
            print("Feature weights applied to KNN model")

        # Set KNN parameters if they exist
        if hasattr(knn_model, 'metric'):
            knn_model.metric = 'hybrid'
            print("KNN metric set to hybrid")
        if hasattr(knn_model, 'weights'):
            knn_model.weights = 'distance'
            print("KNN weights set to distance-based")

        # Print model performance if available
        if metadata:
            print(f"\nModel Performance Summary:")
            rf_perf = metadata.get('rf_performance', {})
            knn_perf = metadata.get('knn_performance', {})
            ensemble_perf = metadata.get('ensemble_performance', {})
            
            if rf_perf:
                print(f"  Random Forest R²: {rf_perf.get('test_r2', 'N/A'):.4f}")
            if knn_perf:
                print(f"  KNN R²: {knn_perf.get('test_r2', 'N/A'):.4f}")
            if ensemble_perf:
                print(f"  Ensemble R²: {ensemble_perf.get('test_r2', 'N/A'):.4f}")

        print("Enhanced models loaded and configured successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Using default values.")
        kmeans_model = None

def calculate_ppi(resolution, screen_size):
    """Calculate PPI from resolution (e.g., '1920x1080') and screen size (inches)."""
    try:
        width, height = map(int, resolution.split('x'))
        diagonal_pixels = np.sqrt(width**2 + height**2)
        return diagonal_pixels / float(screen_size)
    except Exception as e:
        raise ValueError(f"Invalid resolution or screen size: {str(e)}")

def debug_recommendations(X_input, df, preprocessor, knn_model, top_n=5):
    """Debug function to help understand recommendation process"""
    print("\n=== DEBUG: Recommendation Process ===")
    print(f"Input shape: {X_input.shape}")
    print(f"Dataset shape: {df.shape}")
    
    # Get training data
    X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
    if hasattr(X_train_transformed, 'toarray'):
        X_train_transformed = X_train_transformed.toarray()
    print(f"Training data shape: {X_train_transformed.shape}")
    
    # Calculate similarities
    similarities = []
    for i in range(min(10, len(X_train_transformed))):  # Test first 10
        sample = X_train_transformed[i]
        
        # Try different similarity methods
        dot_product = np.dot(X_input[0], sample)
        norm_a = np.linalg.norm(X_input[0])
        norm_b = np.linalg.norm(sample)
        basic_cosine = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
        
        similarities.append((basic_cosine, i))
        print(f"Sample {i}: Cosine similarity = {basic_cosine:.4f}")
    
    # Sort and show top results
    similarities.sort(reverse=True)
    print(f"\nTop 3 similarities:")
    for i, (sim, idx) in enumerate(similarities[:3]):
        laptop = df.iloc[idx]
        print(f"  {i+1}. {laptop.get('Company', 'Unknown')} {laptop.get('TypeName', 'Laptop')} - Similarity: {sim:.4f}")
    
    print("=== END DEBUG ===\n")
    return similarities

# Load models when the server starts
load_models()

# Route definitions
@app.route('/', endpoint='index')
def home():
    return render_template('index.html',
                           companies=companies,
                           types=types,
                           cpus=cpus,
                           gpus=gpus,
                           oss=oss,
                           model_loaded=(preprocessor is not None and rf_model is not None and kmeans_model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    try:
        if preprocessor is None or rf_model is None or knn_model is None or kmeans_model is None:
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

        # Calculate PPI
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

        # Preprocess input for prediction
        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()

        # Price prediction using RandomForest
        prediction = rf_model.predict(X_transformed)
        predicted_price = np.exp(prediction[0])  # Reverse log transformation
        formatted_price = f"₹{predicted_price:,.2f}"  # Fixed: Single currency symbol

        # Enhanced recommendations with price filtering and diversity
        recommendations = []
        if knn_model and hasattr(knn_model, 'get_similar_laptops'):
            recommendations = knn_model.get_similar_laptops(X_transformed, df, top_n=5, price_range_factor=0.3)
        else:
            # Add debug information
            print("Using fallback KNN method")
            debug_recommendations(X_transformed, df, preprocessor, knn_model, top_n=5)
            # Enhanced fallback with price filtering and diversity
            distances = []
            X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
            
            # Calculate similarities more carefully
            for i, sample in enumerate(X_train_transformed):
                # Use simple cosine similarity if hybrid methods aren't available
                if hasattr(knn_model, '_hybrid_similarity') and hasattr(knn_model, 'feature_weights'):
                    sim = knn_model._hybrid_similarity(X_transformed[0], sample, knn_model.feature_weights)
                elif hasattr(knn_model, '_cosine_similarity'):
                    # Use cosine similarity with feature weights if available
                    weights = getattr(knn_model, 'feature_weights', None)
                    sim = knn_model._cosine_similarity(X_transformed[0], sample, weights)
                else:
                    # Fallback to basic cosine similarity
                    dot_product = np.dot(X_transformed[0], sample)
                    norm_a = np.linalg.norm(X_transformed[0])
                    norm_b = np.linalg.norm(sample)
                    sim = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                
                # Apply price-based boost/penalty
                laptop_price = df.iloc[i]['Price']
                price_factor = 1.0
                if 0.7 * predicted_price <= laptop_price <= 1.3 * predicted_price:
                    price_factor = 1.2
                elif 0.5 * predicted_price <= laptop_price <= 1.5 * predicted_price:
                    price_factor = 1.0
                else:
                    price_factor = 0.8
                
                final_score = sim * price_factor
                distances.append((final_score, i))
            
            # Sort by final score (descending)
            distances.sort(reverse=True)
            seen_companies = set()
            seen_types = set()
            
            for similarity_score, idx in distances:
                if len(recommendations) >= 5:
                    break
                    
                rec = df.iloc[idx].to_dict()
                company = rec.get('Company', 'Unknown')
                type_name = rec.get('TypeName', 'Laptop')
                
                # Ensure diversity
                company_count = sum(1 for r in recommendations if r.get('Company') == company)
                type_key = f"{company}-{type_name}"
                
                if company_count >= 2 or type_key in seen_types:
                    continue
                    
                seen_companies.add(company)
                seen_types.add(type_key)
                
                # Build enhanced recommendation
                storage_parts = []
                if rec.get('SSD', 0) > 0:
                    storage_parts.append(f"{int(rec['SSD'])}GB SSD")
                if rec.get('HDD', 0) > 0:
                    storage_parts.append(f"{int(rec['HDD'])}GB HDD")
                storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                
                features = []
                if rec.get('Touchscreen', 0):
                    features.append('Touchscreen')
                if rec.get('Ips', 0):
                    features.append('IPS Display')
                
                ram_val = rec.get('Ram', 0)
                ssd_val = rec.get('SSD', 0)
                if ram_val >= 16 and ssd_val >= 512:
                    features.append('High Performance')
                elif ram_val >= 8 and ssd_val >= 256:
                    features.append('Mid Performance')
                else:
                    features.append('Basic Performance')
                
                recommendations.append({
                    'Company': company,
                    'TypeName': type_name,
                    'Title': f"{company} {type_name}",
                    'Ram': f"{int(rec.get('Ram', 0))}GB",
                    'Storage': storage,
                    'Cpu_brand': rec.get('Cpu brand', 'Unknown'),
                    'Gpu_brand': rec.get('Gpu brand', 'Unknown'),
                    'Weight': f"{rec.get('Weight', 0):.1f}kg" if rec.get('Weight', 0) > 0 else "Weight N/A",
                    'Price': rec.get('Price', 0),
                    'Similarity': f"{similarity_score:.3f}",  # This is what index.html expects
                    'Features': ', '.join(features) if features else 'Standard Features',
                    'Touchscreen': 'Yes' if rec.get('Touchscreen', 0) else 'No',
                    'Ips': 'Yes' if rec.get('Ips', 0) else 'No',
                    'os': rec.get('os', 'Unknown OS')
                })

        # Enhanced clustering with dynamic naming
        cluster_label = None
        cluster_examples = []
        cluster_name = "Unknown Cluster"
        if kmeans_model:
            cluster_num = kmeans_model.predict(X_transformed)[0]
            
            if hasattr(kmeans_model, 'cluster_names'):
                cluster_name = kmeans_model.cluster_names.get(cluster_num, f"Cluster {cluster_num}")
            else:
                cluster_name = DEFAULT_CLUSTER_NAMES.get(cluster_num, f"Cluster {cluster_num}")
            
        # Enhanced clustering with dynamic naming
        cluster_label = None
        cluster_examples = []
        cluster_name = "Unknown Cluster"
        if kmeans_model:
            cluster_num = kmeans_model.predict(X_transformed)[0]
            
            # Check if the model has the enhanced cluster_names attribute
            if hasattr(kmeans_model, 'cluster_names') and kmeans_model.cluster_names:
                cluster_name = kmeans_model.cluster_names.get(cluster_num, f"Cluster {cluster_num}")
            else:
                # Fallback to default names for older models
                cluster_name = DEFAULT_CLUSTER_NAMES.get(cluster_num, f"Cluster {cluster_num}")
            
            # Try to get enhanced cluster examples
            if hasattr(kmeans_model, 'get_cluster_examples'):
                try:
                    X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                    if hasattr(X_train_transformed, 'toarray'):
                        X_train_transformed = X_train_transformed.toarray()
                    cluster_examples = kmeans_model.get_cluster_examples(cluster_num, df, X_all=X_train_transformed, top_n=5)
                except Exception as e:
                    print(f"Enhanced cluster examples failed: {e}")
                    cluster_examples = []
            
            # Fallback cluster examples if enhanced method fails or doesn't exist
            if not cluster_examples:
                try:
                    X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                    if hasattr(X_train_transformed, 'toarray'):
                        X_train_transformed = X_train_transformed.toarray()
                    
                    cluster_labels = kmeans_model.predict(X_train_transformed)
                    cluster_indices = np.where(cluster_labels == cluster_num)[0]
                    
                    if len(cluster_indices) > 0:
                        # Get diverse examples from the cluster
                        cluster_df = df.iloc[cluster_indices].copy()
                        
                        # Simple diversity scoring
                        cluster_df['score'] = (
                            cluster_df['Ram'] * 0.3 +
                            cluster_df.get('SSD', 0) * 0.0002 +
                            np.random.normal(0, 1, len(cluster_df))  # Add randomness for variety
                        )
                        
                        top_diverse = cluster_df.nlargest(min(5, len(cluster_df)), 'score')
                        
                        for _, example in top_diverse.iterrows():
                            # Build storage info
                            ssd_val = example.get('SSD', 0) if example.get('SSD', 0) not in [None, 'N/A', ''] else 0
                            hdd_val = example.get('HDD', 0) if example.get('HDD', 0) not in [None, 'N/A', ''] else 0
                            
                            storage_parts = []
                            if ssd_val > 0:
                                storage_parts.append(f"{int(ssd_val)}GB SSD")
                            if hdd_val > 0:
                                storage_parts.append(f"{int(hdd_val)}GB HDD")
                            storage = " + ".join(storage_parts) if storage_parts else "Storage info unavailable"
                            
                            # Build features
                            features = []
                            if example.get('Touchscreen', 0) == 1 or example.get('Touchscreen') == 'Yes':
                                features.append('Touchscreen')
                            if example.get('Ips', 0) == 1 or example.get('Ips') == 'Yes':
                                features.append('IPS Display')
                            
                            ram_val = example.get('Ram', 0)
                            if ram_val >= 16:
                                features.append('High Memory')
                            elif ram_val >= 8:
                                features.append('Good Memory')
                            
                            features_text = ', '.join(features) if features else 'Standard Features'
                            
                            # Get CPU, GPU, OS with fallbacks
                            cpu_brand = example.get('Cpu brand', example.get('CPU', example.get('Cpu_brand', 'Unknown CPU')))
                            gpu_brand = example.get('Gpu brand', example.get('GPU', example.get('Gpu_brand', 'Unknown GPU')))
                            os_info = example.get('os', example.get('OS', example.get('OpSys', 'Unknown OS')))
                            
                            cluster_examples.append({
                                'Company': example.get('Company', 'Unknown'),
                                'TypeName': example.get('TypeName', 'Laptop'),
                                'Title': f"{example.get('Company', 'Unknown')} {example.get('TypeName', 'Laptop')}",
                                'Ram': f"{int(ram_val)}GB" if ram_val else "N/A",
                                'Storage': storage,
                                'Cpu_brand': cpu_brand,
                                'Gpu_brand': gpu_brand,
                                'Weight': f"{example.get('Weight', 0):.1f}kg" if example.get('Weight', 0) > 0 else "Weight N/A",
                                'Price': f"₹{example.get('Price', 0):,.2f}",
                                'Features': features_text,
                                'Touchscreen': 'Yes' if (example.get('Touchscreen', 0) == 1 or example.get('Touchscreen') == 'Yes') else 'No',
                                'Ips': 'Yes' if (example.get('Ips', 0) == 1 or example.get('Ips') == 'Yes') else 'No',
                                'os': os_info
                            })
                            
                except Exception as e:
                    print(f"Fallback cluster examples also failed: {e}")
                    cluster_examples = []

        return render_template('index.html',
                               predicted_price=formatted_price,
                               recommendations=recommendations,
                               cluster_name=cluster_name,
                               cluster_examples=cluster_examples,
                               companies=companies,
                               types=types,
                               cpus=cpus,
                               gpus=gpus,
                               oss=oss,
                               form_data=form_data,
                               model_loaded=True,
                               login_url=url_for('login'),
                               admin_login_url=url_for('admin_login'),
                               signup_url=url_for('signup'),
                               _anchor='predict-result')

    except Exception as e:
        error_msg = f"Prediction or clustering failed: {str(e)}"
        print(error_msg)
        return render_template('index.html',
                               error=error_msg,
                               companies=companies,
                               types=types,
                               cpus=cpus,
                               gpus=gpus,
                               oss=oss,
                               form_data=form_data,
                               model_loaded=(preprocessor is not None and rf_model is not None and kmeans_model is not None),
                               login_url=url_for('login'),
                               admin_login_url=url_for('admin_login'),
                               signup_url=url_for('signup'),
                               _anchor='predict-result')

@app.route('/predict_history', methods=['POST'])
def predict_history():
    form_data = request.form.to_dict()

    try:
        if preprocessor is None or rf_model is None or knn_model is None or kmeans_model is None:
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

        # Calculate PPI
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

        # Preprocess input for prediction
        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()

        # Price prediction using RandomForest
        prediction = rf_model.predict(X_transformed)
        predicted_price = np.exp(prediction[0])
        formatted_price = f"₹{predicted_price:,.2f}"

        # Enhanced recommendations with same logic as main predict
        recommendations = []
        
        if knn_model and hasattr(knn_model, 'get_similar_laptops'):
            try:
                recommendations = knn_model.get_similar_laptops(X_transformed, df, top_n=5, price_range_factor=0.3)
                for rec in recommendations:
                    if 'similarity_score' not in rec:
                        rec['similarity_score'] = rec.get('Similarity', 0.5)
            except Exception as e:
                print(f"Enhanced KNN failed in predict_history: {e}")
                recommendations = []
        
        if not recommendations:
            # Enhanced fallback with same improvements as main predict route
            distances = []
            X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
            
            for sample_idx, sample in enumerate(X_train_transformed):
                # Use basic cosine similarity
                dot_product = np.dot(X_transformed[0], sample)
                norm_a = np.linalg.norm(X_transformed[0])
                norm_b = np.linalg.norm(sample)
                
                if norm_a > 0 and norm_b > 0:
                    sim = dot_product / (norm_a * norm_b)
                else:
                    sim = 0
                
                # Price consideration
                laptop_price = df.iloc[sample_idx]['Price']
                price_factor = 1.3 if 0.7 * predicted_price <= laptop_price <= 1.3 * predicted_price else 1.0
                distances.append((sim * price_factor, sample_idx))
            
            distances.sort(reverse=True)
            seen_companies = set()
            
            for similarity_score, idx in distances:
                if len(recommendations) >= 5:
                    break
                    
                rec = df.iloc[idx].to_dict()
                company = rec.get('Company', 'Unknown')
                
                if len([r for r in recommendations if r.get('Company') == company]) >= 2:
                    continue
                
                seen_companies.add(company)
                
                # Better data extraction like in main predict route
                ssd_val = rec.get('SSD', 0) if rec.get('SSD', 0) not in [None, 'N/A', ''] else 0
                hdd_val = rec.get('HDD', 0) if rec.get('HDD', 0) not in [None, 'N/A', ''] else 0
                
                storage_parts = []
                if ssd_val > 0:
                    storage_parts.append(f"{int(ssd_val)}GB SSD")
                if hdd_val > 0:
                    storage_parts.append(f"{int(hdd_val)}GB HDD")
                storage = " + ".join(storage_parts) if storage_parts else "Storage info unavailable"
                
                # Handle CPU and GPU properly
                cpu_brand = rec.get('Cpu brand', rec.get('CPU', rec.get('Cpu_brand', 'Unknown CPU')))
                gpu_brand = rec.get('Gpu brand', rec.get('GPU', rec.get('Gpu_brand', 'Unknown GPU')))
                os_info = rec.get('os', rec.get('OS', rec.get('OpSys', 'Unknown OS')))
                
                recommendations.append({
                    'Title': f"{rec.get('Company', 'Unknown')} {rec.get('TypeName', 'Laptop')}",
                    'Company': rec.get('Company', 'Unknown'),
                    'TypeName': rec.get('TypeName', 'Laptop'),
                    'Ram': f"{int(rec.get('Ram', 0))}GB" if rec.get('Ram', 0) else "N/A",
                    'Storage': storage,
                    'Cpu': cpu_brand,  # Template expects 'Cpu' not 'Cpu_brand'
                    'Gpu': gpu_brand,  # Template expects 'Gpu' not 'Gpu_brand'
                    'Weight': f"{rec.get('Weight', 0):.1f}kg" if rec.get('Weight', 0) > 0 else "Weight N/A",
                    'Price': float(rec.get('Price', 0)),
                    'similarity_score': float(similarity_score),  # This should fix the N/A issue
                    'Features': 'Standard Features',
                    'Touchscreen': 'Yes' if (rec.get('Touchscreen', 0) == 1 or rec.get('Touchscreen') == 'Yes') else 'No',
                    'Ips': 'Yes' if (rec.get('Ips', 0) == 1 or rec.get('Ips') == 'Yes') else 'No',
                    'OpSys': os_info,
                    'Inches': screen_size,
                    'resolution': resolution
                })

        # Clustering with backward compatibility
        cluster_label = None
        if kmeans_model:
            cluster_num = kmeans_model.predict(X_transformed)[0]
            # Check if model has enhanced cluster_names, otherwise use default
            if hasattr(kmeans_model, 'cluster_names') and kmeans_model.cluster_names:
                cluster_label = kmeans_model.cluster_names.get(cluster_num, f"Cluster {cluster_num}")
            else:
                cluster_label = DEFAULT_CLUSTER_NAMES.get(cluster_num, f"Cluster {cluster_num}")

        # Save prediction to database if user is logged in
        if 'user_id' in session:
            connection = create_connection()
            if connection:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            """
                            INSERT INTO predictions (
                                uid, company, type, ram, weight, touchscreen, ips,
                                screen_size, resolution, cpu, hdd, ssd, gpu, os, predicted_price
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                session['user_id'], model_data['Company'], model_data['TypeName'],
                                model_data['Ram'], model_data['Weight'], model_data['Touchscreen'],
                                model_data['Ips'], screen_size, resolution, model_data['Cpu brand'],
                                model_data['HDD'], model_data['SSD'], model_data['Gpu brand'],
                                model_data['os'], predicted_price
                            )
                        )
                        connection.commit()
                        flash('Prediction saved successfully!', 'success')
                except Exception as e:
                    print(f"Error saving prediction to database: {str(e)}")
                    flash(f"Prediction completed but couldn't save to history: {str(e)}", 'warning')
                finally:
                    close_connection(connection)

            # Save recommendations to database
            if recommendations:
                connection = create_connection()
                if connection:
                    try:
                        with connection.cursor() as cursor:
                            for rec in recommendations:
                                specs = f"RAM: {rec.get('Ram', 'N/A')}, Storage: {rec.get('Storage', 'N/A')}, CPU: {rec.get('Cpu', 'N/A')}, GPU: {rec.get('Gpu', 'N/A')}"
                                similarity_score = rec.get('similarity_score', 0.5)
                                if isinstance(similarity_score, (int, float)):
                                    similarity_score = float(similarity_score)
                                else:
                                    similarity_score = 0.5
                                    
                                cursor.execute(
                                    """
                                    INSERT INTO recommendations (uid, laptop_name, specs, price, similarity_score)
                                    VALUES (%s, %s, %s, %s, %s)
                                    """,
                                    (session['user_id'], rec['Title'], specs, rec['Price'], similarity_score)
                                )
                            connection.commit()
                    except Exception as e:
                        print(f"Error saving recommendations to database: {str(e)}")
                    finally:
                        close_connection(connection)

        # Get username for template
        username = 'User'
        if 'user_id' in session:
            connection = create_connection()
            if connection:
                try:
                    with connection.cursor(dictionary=True) as cursor:
                        cursor.execute("SELECT username FROM users WHERE uid = %s", (session['user_id'],))
                        user = cursor.fetchone()
                        if user:
                            username = user['username']
                except Exception as e:
                    print(f"Error fetching username: {str(e)}")
                finally:
                    close_connection(connection)

        return render_template('prediction.html',
                              predicted_price=formatted_price,
                              recommendations=recommendations,
                              username=username,
                              form_data=form_data,
                              cluster_label=cluster_label)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)
        flash(error_msg, 'error')
        return redirect(url_for('prediction_history'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        errors = {}

        if not username:
            errors['username'] = "Please enter your username"
        if not password:
            errors['password'] = "Please enter your password"

        if errors:
            return render_template('login.html', errors=errors)

        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                query = "SELECT * FROM users WHERE username = %s"
                cursor.execute(query, (username,))
                user = cursor.fetchone()
                cursor.close()

                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['uid']
                    session['username'] = user['username']
                    flash('Login successful!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    errors['general'] = "Invalid username or password"
                    return render_template('login.html', errors=errors)

            except Exception as e:
                flash(f"An error occurred: {e}", 'error')
                return render_template('login.html', errors=errors)
            finally:
                close_connection(connection)

    return render_template('login.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_username = request.form['username']
        admin_password = request.form['password']
        errors = {}

        if not admin_username:
            errors['username'] = "Please enter the admin username"
        if not admin_password:
            errors['password'] = "Please enter the admin password"

        if errors:
            return render_template('admin login.html', errors=errors)

        ADMIN_CREDENTIALS = {
            'username': 'ayush',
            'password': 'ayush123'
        }

        if admin_username == ADMIN_CREDENTIALS['username'] and admin_password == ADMIN_CREDENTIALS['password']:
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admindashboard'))
        else:
            errors['general'] = "Invalid username or password"
            return render_template('admin login.html', errors=errors)

    return render_template('admin login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    errors = {}
    form_data = {}
    success = False

    if request.method == 'POST':
        try:
            form_data['username'] = request.form.get('username', '')
            form_data['email'] = request.form.get('email', '')
            form_data['password'] = request.form.get('password', '')
            form_data['confirmPassword'] = request.form.get('confirmPassword', '')

            if not form_data['username'] or len(form_data['username']) < 8 or not re.match(r'^[A-Za-z][A-Za-z0-9]{7,19}$', form_data['username']):
                errors['username'] = 'Username must be valid and at least 8 characters long'
            if not form_data['email'] or not re.match(r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$', form_data['email']):
                errors['email'] = 'Invalid email address'
            if not form_data['password'] or len(form_data['password']) < 5 or not re.match(r'^[a-zA-Z0-9]{5,20}$', form_data['password']):
                errors['password'] = 'Password must be at least 5 characters long'
            if not form_data['confirmPassword'] or form_data['password'] != form_data['confirmPassword']:
                errors['confirmPassword'] = 'Passwords do not match'

            if errors:
                return render_template('signup.html', errors=errors, form_data=form_data, success=False)

            connection = create_connection()
            if not connection:
                errors['general'] = 'Database connection error'
                return render_template('signup.html', errors=errors, form_data=form_data, success=False)

            try:
                cursor = connection.cursor()
                cursor.execute("SELECT uid FROM users WHERE username = %s OR email = %s", (form_data['username'], form_data['email']))
                if cursor.fetchone():
                    errors['general'] = 'Username or email already exists'
                    return render_template('signup.html', errors=errors, form_data=form_data, success=False)

                hashed_password = generate_password_hash(form_data['password'], method='pbkdf2:sha256')
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (form_data['username'], form_data['email'], hashed_password)
                )
                connection.commit()
                flash('You have successfully signed up! Please log in.', 'success')
                return render_template('signup.html', errors={}, form_data={}, success=True)
            except Exception as db_error:
                errors['general'] = f'Database error: {str(db_error)}'
                return render_template('signup.html', errors=errors, form_data=form_data, success=False)
            finally:
                cursor.close()
                close_connection(connection)
        except Exception as e:
            errors['general'] = f'Error during signup: {str(e)}'
            return render_template('signup.html', errors=errors, form_data=form_data, success=False)

    return render_template('signup.html', errors={}, form_data={}, success=False)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access dashboard', 'warning')
        return redirect(url_for('login'))

    connection = create_connection()
    user_stats = {
        'total_predictions': 0,
        'average_price': 0,
        'saved_recommendations': 0,
        'total_bookings': 0
    }
    recent_predictions = []
    recent_recommendations = []

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE uid = %s", (session['user_id'],))
            result = cursor.fetchone()
            user_stats['total_predictions'] = result['count'] if result else 0

            cursor.execute("SELECT AVG(predicted_price) as avg_price FROM predictions WHERE uid = %s", (session['user_id'],))
            avg_result = cursor.fetchone()
            user_stats['average_price'] = round(avg_result['avg_price'], 2) if avg_result and avg_result['avg_price'] else 0

            cursor.execute("SELECT COUNT(*) as count FROM recommendations WHERE uid = %s", (session['user_id'],))
            rec_result = cursor.fetchone()
            user_stats['saved_recommendations'] = rec_result['count'] if rec_result else 0

            cursor.execute("SELECT COUNT(*) as count FROM bookings WHERE uid = %s", (session['user_id'],))
            book_result = cursor.fetchone()
            user_stats['total_bookings'] = book_result['count'] if book_result else 0

            cursor.execute("""
                SELECT pid, created_at, predicted_price,
                       CONCAT(COALESCE(company, 'Unknown'), ', ',
                              COALESCE(cpu, 'Unknown'), ', ',
                              COALESCE(ram, 0), 'GB RAM') AS laptop_specs
                FROM predictions
                WHERE uid = %s
                ORDER BY created_at DESC
                LIMIT 5
            """, (session['user_id'],))
            recent_predictions = cursor.fetchall()

            cursor.execute("""
                SELECT laptop_name, price, similarity_score, saved_at
                FROM recommendations
                WHERE uid = %s
                ORDER BY saved_at DESC
                LIMIT 3
            """, (session['user_id'],))
            recent_recommendations = cursor.fetchall()

            cursor.close()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('dashboard.html',
                           companies=companies,
                           types=types,
                           cpus=cpus,
                           gpus=gpus,
                           oss=oss,
                           user_stats=user_stats,
                           recent_predictions=recent_predictions,
                           recent_recommendations=recent_recommendations,
                           model_loaded=(preprocessor is not None and rf_model is not None and kmeans_model is not None))

@app.route('/admindashboard')
def admindashboard():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    users = []
    predictions = []

    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT 10")
                users = cursor.fetchall()
                cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10")
                predictions = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('admindashboard.html', users=users, predictions=predictions)

@app.route('/prediction_history')
def prediction_history():
    if 'user_id' not in session:
        flash('Please log in to view your prediction history.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('dashboard'))

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash('User not found. Please log in again.', 'error')
            session.pop('user_id', None)
            return redirect(url_for('login'))

        username = user['username']

        cursor.execute("""
            SELECT pid, created_at, predicted_price,
                   CONCAT(COALESCE(company, 'Unknown'), ', ',
                          COALESCE(cpu, 'Unknown'), ', ',
                          COALESCE(ram, 0), 'GB RAM, ',
                          COALESCE(ssd, 0), 'GB SSD') AS laptop_specs
            FROM predictions
            WHERE uid = %s
            ORDER BY created_at DESC
        """, (user_id,))
        predictions = cursor.fetchall()

        return render_template('predictionhistory.html', 
                             predictions=predictions, 
                             username=username, 
                             companies=companies, 
                             types=types, 
                             cpus=cpus, 
                             gpus=gpus, 
                             oss=oss)
    except Exception as e:
        flash(f"Error fetching predictions: {str(e)}", 'error')
        return render_template('predictionhistory.html', 
                             predictions=[], 
                             username='User', 
                             companies=companies, 
                             types=types, 
                             cpus=cpus, 
                             gpus=gpus, 
                             oss=oss)
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/view_prediction/<int:pid>')
def view_prediction(pid):
    if 'user_id' not in session:
        flash('Please log in to view prediction details.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('prediction_history'))

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM predictions WHERE pid = %s AND uid = %s
        """, (pid, user_id))
        prediction = cursor.fetchone()
        if not prediction:
            flash('Prediction not found or you do not have permission to view it.', 'error')
            return redirect(url_for('prediction_history'))

        form_data = {
            'company': prediction['company'],
            'type': prediction['type'],
            'ram': prediction['ram'],
            'weight': prediction['weight'],
            'touchscreen': 'Yes' if prediction['touchscreen'] == 1 else 'No',
            'ips': 'Yes' if prediction['ips'] == 1 else 'No',
            'screen_size': prediction['screen_size'],
            'resolution': prediction['resolution'],
            'cpu': prediction['cpu'],
            'HDD': prediction['hdd'],
            'SSD': prediction['ssd'],
            'gpu': prediction['gpu'],
            'os': prediction['os']
        }

        cursor.execute("""
            SELECT laptop_name, specs, price, similarity_score
            FROM recommendations
            WHERE uid = %s
            ORDER BY saved_at DESC
            LIMIT 5
        """, (user_id,))
        recommendations_raw = cursor.fetchall()

        recommendations = []
        for rec in recommendations_raw:
            try:
                specs = rec['specs'] or ""
                company = "Unknown"
                ram = "N/A"
                cpu = "Unknown"
                gpu = "Unknown"
                storage = "N/A"
                
                if "RAM:" in specs:
                    try:
                        ram = specs.split("RAM:")[1].split(",")[0].strip()
                    except:
                        ram = "N/A"
                
                if "CPU:" in specs:
                    try:
                        cpu = specs.split("CPU:")[1].split(",")[0].strip()
                    except:
                        cpu = "Unknown"
                
                if "GPU:" in specs:
                    try:
                        gpu = specs.split("GPU:")[1].split(",")[0].strip()
                    except:
                        gpu = "Unknown"
                
                if "Storage:" in specs:
                    try:
                        storage = specs.split("Storage:")[1].split(",")[0].strip()
                    except:
                        storage = "N/A"

                recommendations.append({
                    'Title': rec['laptop_name'],
                    'Company': company,
                    'TypeName': "Laptop",
                    'Ram': ram,
                    'Storage': storage,
                    'Cpu': cpu,
                    'Gpu': gpu,
                    'Weight': "N/A",
                    'Price': float(rec['price']) if rec['price'] else 0,
                    'similarity_score': float(rec['similarity_score']) if rec['similarity_score'] else 0,
                    'OpSys': "Unknown",
                    'Inches': form_data['screen_size'],
                    'resolution': form_data['resolution'],
                    'Features': "Standard Features"
                })
            except Exception as e:
                recommendations.append({
                    'Title': rec['laptop_name'],
                    'Company': "Unknown",
                    'TypeName': "Laptop",
                    'Ram': "N/A",
                    'Storage': "N/A",
                    'Cpu': "Unknown",
                    'Gpu': "Unknown",
                    'Weight': "N/A",
                    'Price': float(rec['price']) if rec['price'] else 0,
                    'similarity_score': float(rec['similarity_score']) if rec['similarity_score'] else 0,
                    'OpSys': "Unknown",
                    'Inches': form_data['screen_size'],
                    'resolution': form_data['resolution'],
                    'Features': "Standard Features"
                })

        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        username = user['username'] if user else 'User'

        return render_template('prediction.html',
                               predicted_price=f"₹{prediction['predicted_price']:,.2f}",
                               recommendations=recommendations,
                               username=username,
                               form_data=form_data,
                               cluster_label=None)
    except Exception as e:
        flash(f"Error fetching prediction page: {str(e)}", 'error')
        return redirect(url_for('prediction_history'))
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/delete_prediction/<int:pid>', methods=['POST'])
def delete_prediction(pid):
    if 'user_id' not in session:
        flash('Please log in to delete predictions.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('prediction_history'))

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pid FROM predictions WHERE pid = %s AND uid = %s", (pid, user_id))
            if not cursor.fetchone():
                flash('Prediction not found or you do not have permission to delete it.', 'error')
                return redirect(url_for('prediction_history'))

            cursor.execute("DELETE FROM predictions WHERE pid = %s", (pid,))
            connection.commit()
            flash('Prediction deleted successfully.', 'success')
    except Exception as e:
        flash(f"Error deleting prediction: {str(e)}", 'error')
    finally:
        close_connection(connection)

    return redirect(url_for('prediction_history'))

@app.route('/recommendation_history')
def recommendation_history():
    if 'user_id' not in session:
        flash('Please log in to view your recommendation history.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('dashboard'))

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash('User not found. Please log in again.', 'error')
            session.pop('user_id', None)
            return redirect(url_for('login'))

        username = user['username']

        cursor.execute("""
            SELECT rid, laptop_name, specs, price, similarity_score, saved_at
            FROM recommendations
            WHERE uid = %s
            ORDER BY saved_at DESC
        """, (user_id,))
        recommendations = cursor.fetchall()

        return render_template('recommendationhistory.html', recommendations=recommendations, username=username)
    except Exception as e:
        flash(f"Error fetching recommendations: {str(e)}", 'error')
        return render_template('recommendationhistory.html', recommendations=[], username='User')
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/delete_recommendation/<int:rid>', methods=['POST'])
def delete_recommendation(rid):
    if 'user_id' not in session:
        flash('Please log in to delete recommendations.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('recommendation_history'))

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT rid FROM recommendations WHERE rid = %s AND uid = %s", (rid, user_id))
            if not cursor.fetchone():
                flash('Recommendation not found or you do not have permission to delete it.', 'error')
                return redirect(url_for('recommendation_history'))

            cursor.execute("DELETE FROM recommendations WHERE rid = %s", (rid,))
            connection.commit()
            flash('Recommendation deleted successfully.', 'success')
    except Exception as e:
        flash(f"Error deleting recommendation: {str(e)}", 'error')
    finally:
        close_connection(connection)

    return redirect(url_for('recommendation_history'))

@app.route('/booking_history')
def booking_history():
    if 'user_id' not in session:
        flash('Please log in to view your booking history.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('dashboard'))

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash('User not found. Please log in again.', 'error')
            session.pop('user_id', None)
            return redirect(url_for('login'))

        username = user['username']

        cursor.execute("""
            SELECT bid, laptop_name, specs, price, booking_date
            FROM bookings
            WHERE uid = %s
            ORDER BY booking_date DESC
        """, (user_id,))
        bookings = cursor.fetchall()

        return render_template('bookinghistory.html', bookings=bookings, username=username)
    except Exception as e:
        flash(f"Error fetching bookings: {str(e)}", 'error')
        return render_template('bookinghistory.html', bookings=[], username='User')
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('admin_logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)