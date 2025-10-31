import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
from customModel import DecisionTree, RandomForest, CustomKNN, CustomKMeans
from db_connection import create_connection, close_connection
import datetime, json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_in_production'

# Default dropdown options
DEFAULT_COMPANIES = ['Dell', 'HP', 'Lenovo', 'Asus', 'Apple', 'Acer', 'MSI', 'Toshiba', 'Huawei', 'Microsoft']
DEFAULT_TYPES = ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook']
DEFAULT_CPUS = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel Processor', 'AMD Processor']
DEFAULT_GPUS = ['Intel', 'AMD', 'Nvidia']
DEFAULT_OSS = ['Windows', 'Mac', 'Others/No OS/Linux']

# Default cluster names
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
        model_path = r'C:\Users\Wrecker\Desktop\project 2.0 - test\jupyter\laptop_models_full_custom.pkl'
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

        # FIX: Initialize feature weights for KNN model to prevent the warning
        if knn_model and hasattr(knn_model, 'set_feature_weights'):
            # Set default feature weights - adjust based on your feature importance
            n_features = preprocessor.transform(df.head(1).drop(columns=['Price'])).shape[1]
            default_weights = np.ones(n_features)
            knn_model.set_feature_weights(default_weights)
            print(f"DEBUG: Feature weights initialized for KNN model with {n_features} features")

        companies = sorted(df['Company'].unique().tolist())
        types = sorted(df['TypeName'].unique().tolist())
        cpus = sorted(df['Cpu brand'].unique().tolist()) if 'Cpu brand' in df.columns else DEFAULT_CPUS
        gpus = sorted(df['Gpu brand'].unique().tolist()) if 'Gpu brand' in df.columns else DEFAULT_GPUS
        oss = sorted(df['os'].unique().tolist()) if 'os' in df.columns else DEFAULT_OSS

        print("Enhanced models loaded and configured successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Using default values.")
        kmeans_model = None

def calculate_ppi(resolution, screen_size):
    """Calculate PPI from resolution and screen size with robust error handling."""
    try:
        if isinstance(screen_size, str):
            screen_size = float(screen_size)
        
        if resolution and 'x' in resolution:
            width, height = map(int, resolution.split('x'))
            diagonal_pixels = np.sqrt(width**2 + height**2)
            result = diagonal_pixels / screen_size
            return float(result)
        else:
            return 141.0
    except Exception as e:
        print(f"PPI calculation warning: {e}, using default")
        return 141.0

def convert_form_data_types(form_data):
    """Convert form data strings to proper data types for model prediction"""
    converted_data = form_data.copy()
    
    # Define numeric fields and their conversion functions
    numeric_fields = {
        'ram': int,
        'weight': float,
        'screen_size': float,
        'HDD': int,
        'SSD': int
    }
    
    # Boolean fields
    boolean_fields = {
        'touchscreen': lambda x: 1 if x.lower() in ['yes', 'true', '1', 'on'] else 0,
        'ips': lambda x: 1 if x.lower() in ['yes', 'true', '1', 'on'] else 0
    }
    
    # Convert numeric fields
    for field, conv_func in numeric_fields.items():
        if field in converted_data and converted_data[field]:
            try:
                converted_data[field] = conv_func(converted_data[field])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {field}: {converted_data[field]} - {str(e)}")
    
    # Convert boolean fields
    for field, conv_func in boolean_fields.items():
        if field in converted_data:
            converted_data[field] = conv_func(converted_data[field])
    
    return converted_data

def create_engineered_features(input_df):
    """Create the missing engineered features that the preprocessor expects"""
    # Create a copy to avoid modifying the original
    df_engineered = input_df.copy()
    
    # Calculate storage_total
    df_engineered['storage_total'] = df_engineered['HDD'] + df_engineered['SSD']
    
    # Calculate ssd_ratio (avoid division by zero)
    df_engineered['ssd_ratio'] = df_engineered['SSD'] / df_engineered['storage_total'].replace(0, 1)
    
    # Set price_per_ram to 0 (placeholder - this will be calculated after prediction)
    df_engineered['price_per_ram'] = 0
    
    # Determine if it's a gaming laptop
    df_engineered['is_gaming'] = (df_engineered['Gpu brand'].str.contains('NVIDIA', case=False, na=False)).astype(int)
    
    # Determine if it's an ultrabook
    df_engineered['is_ultrabook'] = (df_engineered['TypeName'].str.contains('Ultrabook', case=False, na=False)).astype(int)
    
    print(f"DEBUG: Engineered features created:")
    print(f"  storage_total: {df_engineered['storage_total'].iloc[0]}")
    print(f"  ssd_ratio: {df_engineered['ssd_ratio'].iloc[0]}")
    print(f"  price_per_ram: {df_engineered['price_per_ram'].iloc[0]}")
    print(f"  is_gaming: {df_engineered['is_gaming'].iloc[0]}")
    print(f"  is_ultrabook: {df_engineered['is_ultrabook'].iloc[0]}")
    
    return df_engineered

def save_cluster_category(user_id, pid, cluster_num, cluster_name, cluster_description, example_laptops):
    """Save categorical grouping to database with better error handling"""
    
    # CRITICAL FIX: Convert numpy types to Python types FIRST
    if hasattr(cluster_num, 'item'):
        cluster_num = cluster_num.item()
    cluster_num = int(cluster_num)
    
    if hasattr(user_id, 'item'):
        user_id = user_id.item()
    user_id = int(user_id)
    
    if hasattr(pid, 'item'):
        pid = pid.item()
    pid = int(pid)
    
    print(f"DEBUG: Starting save_cluster_category for user {user_id}, prediction {pid}, cluster {cluster_num}")
    print(f"DEBUG: Converted types - user_id: {type(user_id)}, pid: {type(pid)}, cluster_num: {type(cluster_num)}")
    
    connection = create_connection()
    if not connection:
        print("ERROR: Failed to create database connection for cluster category")
        return False
    
    try:
        with connection.cursor(dictionary=True) as cursor:
            # Convert example_laptops to JSON string with proper error handling
            examples_json = None
            if example_laptops:
                try:
                    serializable_examples = []
                    for example in example_laptops:
                        serializable_example = {}
                        for key, value in example.items():
                            # Convert ALL numpy types to Python types
                            if hasattr(value, 'item'):
                                value = value.item()
                            
                            if key == 'Price':
                                try:
                                    if isinstance(value, str) and '₹' in value:
                                        price_str = value.replace('₹', '').replace(',', '').strip()
                                        serializable_example[key] = float(price_str)
                                    else:
                                        serializable_example[key] = float(value) if value is not None else 0.0
                                except (ValueError, TypeError) as e:
                                    print(f"WARNING: Could not convert price '{value}': {e}")
                                    serializable_example[key] = 0.0
                            elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                                serializable_example[key] = int(value)
                            elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                                serializable_example[key] = float(value)
                            elif isinstance(value, np.bool_):
                                serializable_example[key] = bool(value)
                            elif pd.isna(value):
                                serializable_example[key] = None
                            elif value is None:
                                serializable_example[key] = None
                            else:
                                serializable_example[key] = str(value) if value is not None else None
                        serializable_examples.append(serializable_example)
                    
                    examples_json = json.dumps(serializable_examples, default=str, ensure_ascii=False)
                    print(f"DEBUG: Successfully serialized {len(serializable_examples)} examples to JSON")
                    
                except Exception as e:
                    print(f"ERROR: Error serializing examples to JSON: {e}")
                    import traceback
                    traceback.print_exc()
                    examples_json = json.dumps([])
            else:
                examples_json = json.dumps([])
                print("DEBUG: No example laptops provided, using empty array")
            
            # Check if prediction exists
            cursor.execute("SELECT pid FROM predictions WHERE pid = %s", (pid,))
            prediction_exists = cursor.fetchone()
            if not prediction_exists:
                print(f"ERROR: Prediction with PID {pid} does not exist in database!")
                return False
            else:
                print(f"DEBUG: Prediction with PID {pid} exists in database")
            
            # Check if cluster already exists for this prediction
            cursor.execute(
                "SELECT cid FROM cluster_categories WHERE uid = %s AND pid = %s AND cluster_number = %s",
                (user_id, pid, cluster_num)
            )
            existing_category = cursor.fetchone()
            
            if existing_category:
                # Update existing category
                cursor.execute("""
                    UPDATE cluster_categories 
                    SET cluster_name = %s, cluster_description = %s, example_laptops = %s, created_at = NOW()
                    WHERE cid = %s
                """, (cluster_name, cluster_description, examples_json, existing_category['cid']))
                print(f"DEBUG: Updated existing cluster category: {cluster_name} (CID: {existing_category['cid']})")
            else:
                # Insert new category
                cursor.execute("""
                    INSERT INTO cluster_categories (uid, pid, cluster_number, cluster_name, cluster_description, example_laptops)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (user_id, pid, cluster_num, cluster_name, cluster_description, examples_json))
                new_cid = cursor.lastrowid
                print(f"DEBUG: Created new cluster category: {cluster_name} (CID: {new_cid}) for user {user_id}, prediction {pid}")
            
            connection.commit()
            print(f"SUCCESS: Successfully saved cluster category to database: {cluster_name}")
            return True
            
    except Exception as e:
        print(f"ERROR: Error saving cluster category to database: {e}")
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR DETAILS: {error_details}")
        connection.rollback()
        return False
    finally:
        close_connection(connection)

# Load models when the server starts
load_models()

# =============================================================================
# ROUTE DEFINITIONS
# =============================================================================

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
    print(f"DEBUG: Raw form data received: {form_data}")

    try:
        if preprocessor is None or rf_model is None or knn_model is None or kmeans_model is None:
            raise Exception("Model not loaded properly. Please check the model files.")

        # Convert form data types
        try:
            form_data = convert_form_data_types(form_data)
            print(f"DEBUG: Converted form data: {form_data}")
        except ValueError as e:
            raise ValueError(f"Data conversion error: {str(e)}")

        # Calculate PPI with proper types
        resolution = form_data.get('resolution', '1920x1080')
        screen_size = form_data.get('screen_size', 15.6)
        ppi = calculate_ppi(resolution, screen_size)
        print(f"DEBUG: Calculated PPI: {ppi}")

        # Create model data with proper types
        model_data = {
            'Company': str(form_data.get('company', 'Dell')),
            'TypeName': str(form_data.get('type', 'Notebook')),
            'Ram': int(form_data.get('ram', 8)),
            'Weight': float(form_data.get('weight', 2.0)),
            'Touchscreen': int(form_data.get('touchscreen', 0)),
            'Ips': int(form_data.get('ips', 0)),
            'Cpu brand': str(form_data.get('cpu', 'Intel Core i5')),
            'Gpu brand': str(form_data.get('gpu', 'Intel')),
            'HDD': int(form_data.get('HDD', 0)),
            'SSD': int(form_data.get('SSD', 256)),
            'os': str(form_data.get('os', 'Windows')),
            'ppi': float(ppi)
        }

        print(f"DEBUG: Model data with types:")
        for key, value in model_data.items():
            print(f"  {key}: {value} ({type(value).__name__})")

        # Create DataFrame
        input_df = pd.DataFrame([model_data])
        
        # Explicitly enforce data types
        dtype_mapping = {
            'Ram': 'int64',
            'Weight': 'float64',
            'Touchscreen': 'int64',
            'Ips': 'int64',
            'HDD': 'int64',
            'SSD': 'int64',
            'ppi': 'float64'
        }
        
        for col, dtype in dtype_mapping.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(dtype)
        
        print(f"DEBUG: DataFrame dtypes after enforcement:")
        print(input_df.dtypes)

        # CREATE ENGINEERED FEATURES THAT THE PREPROCESSOR EXPECTS
        input_df = create_engineered_features(input_df)
        
        print(f"DEBUG: DataFrame with engineered features:")
        print(input_df)
        print(f"DEBUG: Final DataFrame columns: {input_df.columns.tolist()}")
        print(f"DEBUG: Final DataFrame dtypes:")
        print(input_df.dtypes)

        # Server-side validation
        if input_df[['Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']].isna().any().any():
            raise ValueError("Invalid input: Please ensure all numeric fields are valid numbers.")
        
        weight_val = float(input_df['Weight'].iloc[0])
        if weight_val < 1 or weight_val > 4:
            raise ValueError("Weight must be between 1 kg and 4 kg.")
        
        ppi_val = float(input_df['ppi'].iloc[0])
        if ppi_val < 100 or ppi_val > 500:
            raise ValueError("Calculated PPI must be between 100 and 500.")
        
        hdd_val = int(input_df['HDD'].iloc[0])
        ssd_val = int(input_df['SSD'].iloc[0])
        if hdd_val == 0 and ssd_val == 0:
            raise ValueError("You must select a non-zero value for either HDD or SSD.")

        # Preprocess input for prediction
        try:
            X_transformed = preprocessor.transform(input_df)
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            print(f"DEBUG: Transformation successful - shape: {X_transformed.shape}, dtype: {X_transformed.dtype}")
        except Exception as e:
            print(f"ERROR: Preprocessing failed: {str(e)}")
            raise ValueError(f"Data preprocessing error: {str(e)}")

        # Price prediction using RandomForest
        try:
            prediction = rf_model.predict(X_transformed)
            predicted_price = np.exp(prediction[0])  # Reverse log transformation
            formatted_price = f"₹{predicted_price:,.2f}"
            print(f"DEBUG: Predicted price: {predicted_price}")
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

        # 🚫 NO DATABASE STORAGE FOR PREDICTIONS FROM INDEX PAGE
        print("DEBUG: Prediction from index page - NOT saving to database")

        # Enhanced recommendations with FIXED KNN feature weights
        recommendations = []
        if knn_model and hasattr(knn_model, 'get_similar_laptops'):
            try:
                # Ensure feature_weights is properly set before calling get_similar_laptops
                if not hasattr(knn_model, 'feature_weights') or knn_model.feature_weights is None:
                    n_features = X_transformed.shape[1]
                    default_weights = np.ones(n_features)
                    if hasattr(knn_model, 'set_feature_weights'):
                        knn_model.set_feature_weights(default_weights)
                        print(f"DEBUG: Set default feature weights for KNN: {default_weights.shape}")
                
                recommendations = knn_model.get_similar_laptops(X_transformed, df, top_n=5, price_range_factor=0.3)
                print(f"DEBUG: Enhanced KNN recommendations found: {len(recommendations)}")
            except Exception as e:
                print(f"WARNING: Enhanced KNN failed: {e}, using fallback")
                recommendations = []

        # Fallback recommendation logic if enhanced method fails
        if not recommendations:
            print("DEBUG: Using fallback recommendation logic")
            X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
            
            distances = []
            for i, sample in enumerate(X_train_transformed):
                # Use simple cosine similarity
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
            
            for similarity_score, idx in distances:
                if len(recommendations) >= 5:
                    break
                    
                rec = df.iloc[idx].to_dict()
                company = rec.get('Company', 'Unknown')
                
                # Ensure diversity
                company_count = sum(1 for r in recommendations if r.get('Company') == company)
                if company_count >= 2:
                    continue
                    
                seen_companies.add(company)
                
                # Build enhanced recommendation
                storage_parts = []
                if rec.get('SSD', 0) > 0:
                    storage_parts.append(f"{int(rec['SSD'])}GB SSD")
                if rec.get('HDD', 0) > 0:
                    storage_parts.append(f"{int(rec['HDD'])}GB HDD")
                storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                
                recommendations.append({
                    'Company': company,
                    'TypeName': rec.get('TypeName', 'Laptop'),
                    'Title': f"{company} {rec.get('TypeName', 'Laptop')}",
                    'Ram': f"{int(rec.get('Ram', 0))}GB",
                    'Storage': storage,
                    'Cpu_brand': rec.get('Cpu brand', 'Unknown'),
                    'Gpu_brand': rec.get('Gpu brand', 'Unknown'),
                    'Weight': f"{rec.get('Weight', 0):.1f}kg" if rec.get('Weight', 0) > 0 else "Weight N/A",
                    'Price': rec.get('Price', 0),
                    'Similarity': f"{similarity_score:.3f}",
                    'Touchscreen': 'Yes' if rec.get('Touchscreen', 0) else 'No',
                    'Ips': 'Yes' if rec.get('Ips', 0) else 'No',
                    'os': rec.get('os', 'Unknown OS')
                })

        # 🚫 NO DATABASE STORAGE FOR RECOMMENDATIONS FROM INDEX PAGE
        print("DEBUG: Recommendations from index page - NOT saving to database")

        # Enhanced clustering with dynamic naming
        cluster_label = None
        cluster_examples = []
        cluster_name = "Unknown Cluster"
        if kmeans_model:
            try:
                cluster_num = kmeans_model.predict(X_transformed)[0]
                print(f"DEBUG: Predicted cluster: {cluster_num}")
                
                if hasattr(kmeans_model, 'cluster_names'):
                    cluster_name = kmeans_model.cluster_names.get(cluster_num, f"Cluster {cluster_num}")
                else:
                    cluster_name = DEFAULT_CLUSTER_NAMES.get(cluster_num, f"Cluster {cluster_num}")
                
                cluster_label = cluster_name

                # Get cluster examples
                try:
                    X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                    if hasattr(X_train_transformed, 'toarray'):
                        X_train_transformed = X_train_transformed.toarray()
                    
                    cluster_labels = kmeans_model.predict(X_train_transformed)
                    cluster_indices = np.where(cluster_labels == cluster_num)[0]
                    
                    if len(cluster_indices) > 0:
                        cluster_df = df.iloc[cluster_indices].copy()
                        cluster_df['score'] = (
                            cluster_df['Ram'] * 0.3 +
                            cluster_df.get('SSD', 0) * 0.0002 +
                            np.random.normal(0, 1, len(cluster_df))
                        )
                        
                        top_diverse = cluster_df.nlargest(min(5, len(cluster_df)), 'score')
                        
                        for _, example in top_diverse.iterrows():
                            ssd_val = example.get('SSD', 0) if example.get('SSD', 0) not in [None, 'N/A', ''] else 0
                            hdd_val = example.get('HDD', 0) if example.get('HDD', 0) not in [None, 'N/A', ''] else 0
                            
                            storage_parts = []
                            if ssd_val > 0:
                                storage_parts.append(f"{int(ssd_val)}GB SSD")
                            if hdd_val > 0:
                                storage_parts.append(f"{int(hdd_val)}GB HDD")
                            storage = " + ".join(storage_parts) if storage_parts else "Storage info unavailable"
                            
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
                        print(f"DEBUG: Found {len(cluster_examples)} cluster examples")
                        
                        # 🚫 NO DATABASE STORAGE FOR CLUSTER DATA FROM INDEX PAGE
                        print("DEBUG: Cluster data from index page - NOT saving to database")
                            
                except Exception as e:
                    print(f"WARNING: Cluster examples failed: {e}")
                    cluster_examples = []
                    
            except Exception as e:
                print(f"WARNING: Clustering failed: {e}")
                cluster_name = "Clustering Unavailable"

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
                               model_loaded=True)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                               error=error_msg,
                               companies=companies,
                               types=types,
                               cpus=cpus,
                               gpus=gpus,
                               oss=oss,
                               form_data=form_data,
                               model_loaded=(preprocessor is not None and rf_model is not None and kmeans_model is not None))

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

@app.route('/dashboard')
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
    username = 'User'

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("SELECT username FROM users WHERE uid = %s", (session['user_id'],))
            user = cursor.fetchone()
            if user:
                username = user['username']
            else:
                flash('User not found. Please log in again.', 'error')
                session.pop('user_id', None)
                return redirect(url_for('login'))

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
                           username=username,
                           user_stats=user_stats,
                           recent_predictions=recent_predictions,
                           recent_recommendations=recent_recommendations,
                           model_loaded=(preprocessor is not None and rf_model is not None and kmeans_model is not None))


# =============================================================================
# RECOMMENDATION HISTORY ROUTES
# =============================================================================

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
        
        # Get username
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash('User not found. Please log in again.', 'error')
            session.pop('user_id', None)
            return redirect(url_for('login'))

        username = user['username']

        # Get recommendations grouped by prediction with prediction details
        cursor.execute("""
            SELECT 
                r.rid, 
                r.laptop_name, 
                r.specs, 
                r.price, 
                r.similarity_score, 
                r.saved_at,
                r.pid,
                p.pid as prediction_id,
                p.created_at as prediction_date,
                p.company,
                p.type,
                p.ram,
                p.cpu,
                p.predicted_price,
                CONCAT(COALESCE(p.company, 'Unknown'), ' ', 
                       COALESCE(p.type, 'Laptop'), ' - ', 
                       COALESCE(p.ram, 0), 'GB RAM, ', 
                       COALESCE(p.cpu, 'Unknown CPU')) as search_criteria
            FROM recommendations r
            LEFT JOIN predictions p ON r.pid = p.pid
            WHERE r.uid = %s
            ORDER BY p.created_at DESC, r.saved_at DESC
        """, (user_id,))
        
        recommendations_raw = cursor.fetchall()

        # Group recommendations by prediction
        prediction_groups = {}
        total_recommendations = 0
        all_similarities = []
        all_prices = []

        for rec in recommendations_raw:
            prediction_id = rec['pid'] or 'unknown_' + str(rec['rid'])
            
            if prediction_id not in prediction_groups:
                # Create prediction group
                prediction_groups[prediction_id] = {
                    'prediction_id': rec['prediction_id'],
                    'search_criteria': rec['search_criteria'] or 'Custom Search',
                    'prediction_date': rec['prediction_date'],
                    'predicted_price': rec['predicted_price'],
                    'recommendations': []
                }
            
            # Add recommendation to group
            prediction_groups[prediction_id]['recommendations'].append({
                'rid': rec['rid'],
                'laptop_name': rec['laptop_name'],
                'specs': rec['specs'],
                'price': rec['price'],
                'similarity_score': rec['similarity_score'],
                'saved_at': rec['saved_at']
            })
            
            total_recommendations += 1
            if rec['similarity_score']:
                all_similarities.append(float(rec['similarity_score']))
            if rec['price']:
                all_prices.append(float(rec['price']))

        # Convert to list for template
        prediction_groups_list = list(prediction_groups.values())
        
        # Calculate stats
        average_match = 0
        best_price = 0
        
        if all_similarities:
            average_match = round((sum(all_similarities) / len(all_similarities)) * 100)
        if all_prices:
            best_price = min(all_prices)

        print(f"DEBUG: Found {total_recommendations} recommendations in {len(prediction_groups_list)} prediction groups")

        return render_template('recommendationhistory.html', 
                             prediction_groups=prediction_groups_list,
                             total_recommendations=total_recommendations,
                             average_match=average_match,
                             best_price=best_price,
                             username=username)

    except Exception as e:
        flash(f"Error fetching recommendations: {str(e)}", 'error')
        import traceback
        traceback.print_exc()
        return render_template('recommendationhistory.html', 
                             prediction_groups=[],
                             total_recommendations=0,
                             average_match=0,
                             best_price=0,
                             username='User')
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/book_recommendation/<int:rid>', methods=['POST'])
def book_recommendation(rid):
    if 'user_id' not in session:
        flash('Please login to book a laptop', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('recommendation_history'))

    try:
        cursor = connection.cursor(dictionary=True)
        
        # Fetch recommendation details
        cursor.execute("""
            SELECT laptop_name, specs, price
            FROM recommendations
            WHERE rid = %s AND uid = %s
        """, (rid, user_id))
        recommendation = cursor.fetchone()

        if not recommendation:
            flash('Recommendation not found or you do not have permission to book it.', 'error')
            return redirect(url_for('recommendation_history'))

        # Insert into bookings table with 'pending' status
        cursor.execute("""
            INSERT INTO bookings (uid, rid, laptop_name, specs, price, booking_type, booking_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (user_id, rid, recommendation['laptop_name'], recommendation['specs'], 
              recommendation['price'], 'recommendation', 'pending'))  # Changed to 'pending'
        
        connection.commit()
        flash('Laptop booked successfully! Your booking is pending confirmation.', 'success')
        print(f"DEBUG: Booked recommendation {rid} with status: pending")

    except Exception as e:
        flash(f"An error occurred while booking: {str(e)}", 'error')
        print(f"ERROR booking recommendation: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

    return redirect(url_for('recommendation_history'))

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

# =============================================================================
# CATEGORY HISTORY ROUTES
# =============================================================================

@app.route('/category_history')
def category_history():
    if 'user_id' not in session:
        flash('Please log in to view your category history.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('dashboard'))

    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get username
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            flash('User not found. Please log in again.', 'error')
            session.pop('user_id', None)
            return redirect(url_for('login'))

        username = user['username']

        # Get categories with prediction information
        cursor.execute("""
            SELECT 
                cc.cid, 
                cc.cluster_number, 
                cc.cluster_name, 
                cc.cluster_description, 
                cc.example_laptops, 
                cc.created_at,
                cc.pid,
                p.created_at as prediction_date,
                p.company,
                p.type,
                p.ram,
                p.cpu,
                p.predicted_price,
                CONCAT(COALESCE(p.company, 'Unknown'), ' ', 
                       COALESCE(p.type, 'Laptop'), ' - ', 
                       COALESCE(p.ram, 0), 'GB RAM, ', 
                       COALESCE(p.cpu, 'Unknown CPU')) as prediction_details
            FROM cluster_categories cc
            LEFT JOIN predictions p ON cc.pid = p.pid
            WHERE cc.uid = %s
            ORDER BY cc.created_at DESC
        """, (user_id,))
        
        categories_raw = cursor.fetchall()

        # Process categories and parse JSON
        categories = []
        unique_clusters = set()
        total_examples = 0
        
        for cat in categories_raw:
            try:
                example_laptops = []
                if cat['example_laptops']:
                    try:
                        examples_data = json.loads(cat['example_laptops'])
                        # Process each example laptop and add individual booking data
                        for i, example in enumerate(examples_data):
                            # Generate a unique ID for each laptop for booking
                            laptop_id = f"{cat['cid']}_{i}"
                            
                            # Ensure price is properly formatted
                            price = example.get('Price', 0)
                            numeric_price = 0
                            
                            if isinstance(price, str):
                                if '₹' in price:
                                    try:
                                        numeric_price = float(price.replace('₹', '').replace(',', ''))
                                    except (ValueError, TypeError):
                                        numeric_price = 0
                                else:
                                    try:
                                        numeric_price = float(price)
                                    except (ValueError, TypeError):
                                        numeric_price = 0
                            else:
                                numeric_price = float(price) if price else 0
                            
                            # Add laptop ID and numeric price for individual booking
                            example['laptop_id'] = laptop_id
                            example['numeric_price'] = numeric_price
                            example_laptops.append(example)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Error parsing category {cat['cid']} examples: {e}")
                        example_laptops = []

                # Add prediction information to category
                category_data = {
                    'cid': cat['cid'],
                    'cluster_number': cat['cluster_number'],
                    'cluster_name': cat['cluster_name'],
                    'cluster_description': cat['cluster_description'],
                    'example_laptops': example_laptops,
                    'created_at': cat['created_at'],
                    'prediction_id': cat['pid'],
                    'prediction_date': cat['prediction_date'],
                    'prediction_details': cat['prediction_details'],
                    'predicted_price': cat['predicted_price']
                }
                
                categories.append(category_data)
                unique_clusters.add(cat['cluster_number'])
                total_examples += len(example_laptops)
                
            except Exception as e:
                print(f"Error processing category {cat.get('cid', 'unknown')}: {e}")
                # Add basic category data even if examples fail
                categories.append({
                    'cid': cat['cid'],
                    'cluster_number': cat['cluster_number'],
                    'cluster_name': cat['cluster_name'],
                    'cluster_description': cat['cluster_description'],
                    'example_laptops': [],
                    'created_at': cat['created_at'],
                    'prediction_id': cat['pid'],
                    'prediction_date': cat['prediction_date'],
                    'prediction_details': cat['prediction_details'],
                    'predicted_price': cat['predicted_price']
                })
                unique_clusters.add(cat['cluster_number'])

        print(f"DEBUG: Found {len(categories)} categories with {total_examples} total examples")

        return render_template('categoryhistory.html',
                             categories=categories,
                             username=username,
                             unique_clusters=unique_clusters,
                             total_examples=total_examples)

    except Exception as e:
        flash(f"Error fetching categories: {str(e)}", 'error')
        import traceback
        traceback.print_exc()
        return render_template('categoryhistory.html',
                             categories=[],
                             username='User',
                             unique_clusters=set(),
                             total_examples=0)
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/book_laptop/<laptop_id>/<int:cid>', methods=['POST'])
def book_laptop(laptop_id, cid):
    if 'user_id' not in session:
        flash('Please login to book a laptop', 'warning')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('category_history'))

    try:
        cursor = connection.cursor(dictionary=True)
        
        # Verify the category belongs to the user
        cursor.execute("""
            SELECT cluster_name, example_laptops 
            FROM cluster_categories 
            WHERE cid = %s AND uid = %s
        """, (cid, user_id))
        
        category = cursor.fetchone()
        if not category:
            flash('Category not found or you do not have permission to book from it.', 'error')
            return redirect(url_for('category_history'))

        # Parse example laptops to find the specific laptop
        laptop_details = None
        if category['example_laptops']:
            try:
                examples = json.loads(category['example_laptops'])
                for i, example in enumerate(examples):
                    # Generate the same ID used in the template
                    example_id = f"{cid}_{i}"
                    if example_id == laptop_id:
                        laptop_details = example
                        break
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing examples for booking: {e}")

        if not laptop_details:
            flash('Laptop not found in this category.', 'error')
            return redirect(url_for('category_history'))

        # Extract laptop information
        laptop_name = laptop_details.get('Title') or f"{laptop_details.get('Company', 'Unknown')} {laptop_details.get('TypeName', 'Laptop')}"
        
        # Build specs string
        specs_parts = []
        if laptop_details.get('Ram'):
            specs_parts.append(f"RAM: {laptop_details['Ram']}")
        if laptop_details.get('Storage'):
            specs_parts.append(f"Storage: {laptop_details['Storage']}")
        if laptop_details.get('Cpu_brand'):
            specs_parts.append(f"CPU: {laptop_details['Cpu_brand']}")
        if laptop_details.get('Gpu_brand'):
            specs_parts.append(f"GPU: {laptop_details['Gpu_brand']}")
        
        specs = ', '.join(specs_parts) if specs_parts else 'From category cluster'
        
        # Get price - handle both string and numeric formats
        price = laptop_details.get('numeric_price', 0)
        if not price:
            price_str = laptop_details.get('Price', '0')
            if isinstance(price_str, str):
                if '₹' in price_str:
                    try:
                        price = float(price_str.replace('₹', '').replace(',', ''))
                    except (ValueError, TypeError):
                        price = 0
                else:
                    try:
                        price = float(price_str)
                    except (ValueError, TypeError):
                        price = 0
            else:
                price = float(price_str) if price_str else 0

        # Use 'category' as the booking_type and set status to 'pending'
        cursor.execute("""
            INSERT INTO bookings (uid, laptop_name, specs, price, booking_type, booking_status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, laptop_name, specs, price, 'category', 'pending'))  # Changed to 'pending'
        
        connection.commit()
        flash(f'Laptop "{laptop_name}" booked successfully from category! Your booking is pending confirmation.', 'success')
        print(f"DEBUG: Booked laptop {laptop_id} from category {cid} with booking_type: category, status: pending")

    except Exception as e:
        flash(f"An error occurred while booking: {str(e)}", 'error')
        print(f"ERROR booking laptop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

    return redirect(url_for('category_history'))

@app.route('/delete_category/<int:cid>', methods=['POST'])
def delete_category(cid):
    if 'user_id' not in session:
        flash('Please log in to delete categories.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return redirect(url_for('category_history'))

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT cid FROM cluster_categories WHERE cid = %s AND uid = %s", (cid, user_id))
            if not cursor.fetchone():
                flash('Category not found or you do not have permission to delete it.', 'error')
                return redirect(url_for('category_history'))

            cursor.execute("DELETE FROM cluster_categories WHERE cid = %s", (cid,))
            connection.commit()
            flash('Category deleted successfully.', 'success')
    except Exception as e:
        flash(f"Error deleting category: {str(e)}", 'error')
    finally:
        close_connection(connection)

    return redirect(url_for('category_history'))

# =============================================================================
# BOOKING HISTORY ROUTE
# =============================================================================

@app.route('/booking_history')
def booking_history():
    if 'user_id' not in session:
        flash('Please login to access booking history', 'warning')
        return redirect(url_for('login'))

    connection = create_connection()
    bookings = []
    username = 'User'

    # Initialize statistics
    total_bookings = 0
    total_value = 0
    best_price = 0

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            
            # Fetch username
            cursor.execute("SELECT username FROM users WHERE uid = %s", (session['user_id'],))
            user = cursor.fetchone()
            if user:
                username = user['username']
            else:
                flash('User not found. Please log in again.', 'error')
                session.pop('user_id', None)
                return redirect(url_for('login'))

            # Fetch booking history with ALL fields including booking_status
            cursor.execute("""
                SELECT bid, laptop_name, specs, price, booked_at, booking_type, booking_status, updated_at
                FROM bookings
                WHERE uid = %s
                ORDER BY booked_at DESC
            """, (session['user_id'],))
            bookings_raw = cursor.fetchall()

            # Process bookings and calculate statistics
            valid_prices = []
            
            for booking in bookings_raw:
                booking_type = booking.get('booking_type', 'standard')
                booking_status = booking.get('booking_status', 'pending')
                
                # Process price for statistics
                price = booking.get('price', 0)
                if price is not None:
                    try:
                        # Convert price to float for calculations
                        price_float = float(price)
                        valid_prices.append(price_float)
                        total_value += price_float
                    except (ValueError, TypeError):
                        # If price conversion fails, use 0 for this booking
                        print(f"Warning: Invalid price value '{price}' for booking {booking.get('bid')}")
                
                # Set display properties based on booking type
                if booking_type == 'recommendation':
                    type_label = 'Recommendation'
                    type_icon = 'fa-star'
                    type_class = 'recommendation'
                elif booking_type == 'category':
                    type_label = 'Category'
                    type_icon = 'fa-layer-group'
                    type_class = 'category'
                elif booking_type == 'individual_laptop':
                    type_label = 'Individual'
                    type_icon = 'fa-laptop'
                    type_class = 'individual'
                else:
                    type_label = 'Standard'
                    type_icon = 'fa-shopping-cart'
                    type_class = 'standard'
                
                # Set status display properties
                if booking_status == 'pending':
                    status_label = 'Pending'
                    status_icon = 'fa-clock'
                    status_class = 'pending'
                elif booking_status == 'confirmed':
                    status_label = 'Confirmed'
                    status_icon = 'fa-check-circle'
                    status_class = 'confirmed'
                elif booking_status == 'shipped':
                    status_label = 'Shipped'
                    status_icon = 'fa-shipping-fast'
                    status_class = 'shipped'
                elif booking_status == 'delivered':
                    status_label = 'Delivered'
                    status_icon = 'fa-box-open'
                    status_class = 'delivered'
                elif booking_status == 'cancelled':
                    status_label = 'Cancelled'
                    status_icon = 'fa-times-circle'
                    status_class = 'cancelled'
                else:
                    status_label = booking_status.title()
                    status_icon = 'fa-question-circle'
                    status_class = 'unknown'
                
                booking_data = {
                    'bid': booking['bid'],
                    'laptop_name': booking['laptop_name'],
                    'specs': booking['specs'],
                    'price': price,
                    'booked_at': booking['booked_at'],
                    'updated_at': booking.get('updated_at'),
                    'booking_type': booking_type,
                    'booking_status': booking_status,
                    'type_label': type_label,
                    'type_icon': type_icon,
                    'type_class': type_class,
                    'status_label': status_label,
                    'status_icon': status_icon,
                    'status_class': status_class
                }
                bookings.append(booking_data)

            # Calculate final statistics
            total_bookings = len(bookings)
            
            if valid_prices:
                best_price = min(valid_prices)
            else:
                best_price = 0

            cursor.close()
            print(f"DEBUG: Found {total_bookings} bookings for user {session['user_id']}")
            print(f"DEBUG: Total value: {total_value}, Best price: {best_price}")
            
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            print(f"ERROR in booking_history: {e}")
            import traceback
            traceback.print_exc()
        finally:
            close_connection(connection)

    return render_template('bookinghistory.html',
                           username=username,
                           bookings=bookings,
                           total_bookings=total_bookings,
                           total_value=total_value,
                           best_price=best_price)

@app.route('/delete_booking/<int:bid>', methods=['POST'])
def delete_booking(bid):
    if 'user_id' not in session:
        flash('Please login to delete a booking', 'warning')
        return redirect(url_for('login'))

    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM bookings WHERE bid = %s AND uid = %s", (bid, session['user_id']))
            if cursor.rowcount == 0:
                flash('Booking not found or you do not have permission to delete it.', 'error')
            else:
                connection.commit()
                flash('Booking deleted successfully!', 'success')
            cursor.close()
        except Exception as e:
            flash(f"An error occurred while deleting: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('booking_history'))

# =============================================================================
# PREDICTION HISTORY ROUTE
# =============================================================================

@app.route('/prediction_history', methods=['GET', 'POST'])
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

        if request.method == 'POST':
            try:
                if preprocessor is None or rf_model is None or knn_model is None or kmeans_model is None:
                    raise Exception("Model not loaded properly. Please check the model files.")

                # Get form data
                form_data = request.form.to_dict()
                print(f"DEBUG prediction_history: Raw form data: {form_data}")

                # STEP 1: Convert ALL numeric values FIRST, before creating the DataFrame
                try:
                    # Extract and convert with explicit type casting
                    ram = int(form_data.get('ram', '8'))
                    weight = float(form_data.get('weight', '2.0'))
                    hdd = int(form_data.get('HDD', '0'))
                    ssd = int(form_data.get('SSD', '256'))
                    screen_size = float(form_data.get('screen_size', '15.6'))
                    touchscreen = 1 if form_data.get('touchscreen') == 'Yes' else 0
                    ips = 1 if form_data.get('ips') == 'Yes' else 0
                    
                    # Calculate PPI
                    resolution = form_data.get('resolution', '1920x1080')
                    ppi = float(calculate_ppi(resolution, screen_size))
                    
                    print(f"DEBUG prediction_history: Converted numeric values:")
                    print(f"  RAM: {ram} ({type(ram).__name__})")
                    print(f"  Weight: {weight} ({type(weight).__name__})")
                    print(f"  HDD: {hdd} ({type(hdd).__name__})")
                    print(f"  SSD: {ssd} ({type(ssd).__name__})")
                    print(f"  Screen Size: {screen_size} ({type(screen_size).__name__})")
                    print(f"  Touchscreen: {touchscreen} ({type(touchscreen).__name__})")
                    print(f"  IPS: {ips} ({type(ips).__name__})")
                    print(f"  PPI: {ppi} ({type(ppi).__name__})")
                    
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid numeric input: {str(e)}")

                # STEP 2: Validate ranges
                if ram not in [4, 8, 16, 32, 64]:
                    raise ValueError("Invalid RAM value. Must be 4, 8, 16, 32, or 64 GB")
                if weight < 1 or weight > 4:
                    raise ValueError("Weight must be between 1 kg and 4 kg")
                if ppi < 100 or ppi > 500:
                    raise ValueError("Calculated PPI must be between 100 and 500")
                if hdd == 0 and ssd == 0:
                    raise ValueError("You must select a non-zero value for either HDD or SSD")
                if not all(key in form_data for key in ['company', 'type', 'cpu', 'gpu', 'os']):
                    raise ValueError("Missing required categorical fields")

                # STEP 3: Create DataFrame with ALREADY CONVERTED values
                input_data = {
                    'Company': str(form_data.get('company')),
                    'TypeName': str(form_data.get('type')),
                    'Ram': ram,  # Already int
                    'Weight': weight,  # Already float
                    'Touchscreen': touchscreen,  # Already int
                    'Ips': ips,  # Already int
                    'Cpu brand': str(form_data.get('cpu')),
                    'Gpu brand': str(form_data.get('gpu')),
                    'HDD': hdd,  # Already int
                    'SSD': ssd,  # Already int
                    'os': str(form_data.get('os')),
                    'ppi': ppi  # Already float
                }
                
                print(f"DEBUG prediction_history: Input data with types: {[(k, v, type(v).__name__) for k, v in input_data.items()]}")
                
                # Create DataFrame - pandas should preserve the types we give it
                input_df = pd.DataFrame([input_data])
                
                # STEP 4: CRITICAL - Force numeric dtypes explicitly
                input_df['Ram'] = input_df['Ram'].astype('int64')
                input_df['Weight'] = input_df['Weight'].astype('float64')
                input_df['Touchscreen'] = input_df['Touchscreen'].astype('int64')
                input_df['Ips'] = input_df['Ips'].astype('int64')
                input_df['HDD'] = input_df['HDD'].astype('int64')
                input_df['SSD'] = input_df['SSD'].astype('int64')
                input_df['ppi'] = input_df['ppi'].astype('float64')
                
                print(f"DEBUG prediction_history: DataFrame dtypes AFTER explicit conversion:")
                print(input_df.dtypes)
                print(f"DEBUG prediction_history: DataFrame values:")
                print(input_df)

                # STEP 4.5: CREATE ENGINEERED FEATURES FOR PREDICTION HISTORY
                input_df = create_engineered_features(input_df)

                print(f"DEBUG prediction_history: DataFrame with engineered features:")
                print(input_df)
                print(f"DEBUG prediction_history: Final DataFrame columns: {input_df.columns.tolist()}")
                
                # STEP 5: Transform with preprocessor
                try:
                    X_transformed = preprocessor.transform(input_df)
                    if hasattr(X_transformed, 'toarray'):
                        X_transformed = X_transformed.toarray()
                    
                    print(f"DEBUG prediction_history: Transformation successful!")
                    print(f"  Shape: {X_transformed.shape}")
                    print(f"  Type: {type(X_transformed)}")
                    print(f"  Dtype: {X_transformed.dtype}")
                    
                except Exception as e:
                    print(f"ERROR prediction_history: Preprocessor transformation failed!")
                    print(f"  Error: {str(e)}")
                    print(f"  DataFrame dtypes at time of error: {input_df.dtypes}")
                    raise ValueError(f"Data preprocessing error: {str(e)}")

                # Price prediction
                try:
                    prediction = rf_model.predict(X_transformed)
                    if hasattr(prediction[0], 'item'):
                        prediction_value = prediction[0].item()
                    else:
                        prediction_value = float(prediction[0])
                    
                    predicted_price = np.exp(prediction_value)
                    formatted_price = f"₹{predicted_price:,.2f}"
                    print(f"DEBUG prediction_history: Predicted price: {predicted_price}")
                    
                except Exception as e:
                    raise ValueError(f"Prediction error: {str(e)}")

                # Save prediction to database
                save_connection = create_connection()
                if not save_connection:
                    raise Exception("Database connection failed")

                try:
                    with save_connection.cursor() as cursor_save:
                        cursor_save.execute(
                            """
                            INSERT INTO predictions (
                                uid, company, type, ram, weight, touchscreen, ips,
                                screen_size, resolution, cpu, hdd, ssd, gpu, os, predicted_price
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                user_id, input_data['Company'], input_data['TypeName'],
                                input_data['Ram'], input_data['Weight'], input_data['Touchscreen'],
                                input_data['Ips'], screen_size, resolution, input_data['Cpu brand'],
                                input_data['HDD'], input_data['SSD'], input_data['Gpu brand'],
                                input_data['os'], predicted_price
                            )
                        )
                        pid = cursor_save.lastrowid
                        save_connection.commit()
                        flash('Prediction saved successfully!', 'success')
                        print(f"DEBUG prediction_history: Prediction saved with PID: {pid}")
                except Exception as e:
                    print(f"ERROR prediction_history: Error saving prediction to database: {str(e)}")
                    flash(f"Prediction completed but couldn't save to history: {str(e)}", 'warning')
                    raise
                finally:
                    close_connection(save_connection)

                # FIXED: Enhanced clustering with consistent data structure
                cluster_label = None
                cluster_examples = []
                cluster_name = "Unknown Cluster"
                cluster_num = None
                cluster_description = "No cluster description available"

                if kmeans_model:
                    try:
                        cluster_prediction = kmeans_model.predict(X_transformed)
                        if hasattr(cluster_prediction[0], 'item'):
                            cluster_num = cluster_prediction[0].item()
                        else:
                            cluster_num = int(cluster_prediction[0])
                            
                        print(f"DEBUG prediction_history: Predicted cluster number: {cluster_num}")
                        
                        if hasattr(kmeans_model, 'cluster_names') and kmeans_model.cluster_names:
                            cluster_name = kmeans_model.cluster_names.get(cluster_num, f"Cluster {cluster_num}")
                            cluster_description = f"Laptops with similar specifications and price range (Cluster {cluster_num})"
                        else:
                            cluster_name = DEFAULT_CLUSTER_NAMES.get(cluster_num, f"Cluster {cluster_num}")
                            cluster_description = f"Category of laptops with similar characteristics (Cluster {cluster_num})"
                        
                        cluster_label = cluster_name
                        print(f"DEBUG prediction_history: Cluster name: {cluster_name}")

                        # Get cluster examples - USE THE SAME METHOD AS INDEX PAGE
                        cluster_examples = []
                        try:
                            X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                            if hasattr(X_train_transformed, 'toarray'):
                                X_train_transformed = X_train_transformed.toarray()
                            
                            cluster_labels = kmeans_model.predict(X_train_transformed)
                            cluster_indices = np.where(cluster_labels == cluster_num)[0]
                            
                            if len(cluster_indices) > 0:
                                cluster_df = df.iloc[cluster_indices].copy()
                                cluster_df['score'] = (
                                    cluster_df['Ram'] * 0.3 +
                                    cluster_df.get('SSD', 0) * 0.0002 +
                                    np.random.normal(0, 1, len(cluster_df))
                                )
                                
                                top_diverse = cluster_df.nlargest(min(5, len(cluster_df)), 'score')
                                
                                for _, example in top_diverse.iterrows():
                                    ssd_val = example.get('SSD', 0) if example.get('SSD', 0) not in [None, 'N/A', ''] else 0
                                    hdd_val = example.get('HDD', 0) if example.get('HDD', 0) not in [None, 'N/A', ''] else 0
                                    
                                    storage_parts = []
                                    if ssd_val > 0:
                                        storage_parts.append(f"{int(ssd_val)}GB SSD")
                                    if hdd_val > 0:
                                        storage_parts.append(f"{int(hdd_val)}GB HDD")
                                    storage = " + ".join(storage_parts) if storage_parts else "Storage info unavailable"
                                    
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
                                    
                                    cpu_brand = example.get('Cpu brand', example.get('CPU', example.get('Cpu_brand', 'Unknown CPU')))
                                    gpu_brand = example.get('Gpu brand', example.get('GPU', example.get('Gpu_brand', 'Unknown GPU')))
                                    os_info = example.get('os', example.get('OS', example.get('OpSys', 'Unknown OS')))
                                    
                                    price_value = example.get('Price', 0)
                                    if hasattr(price_value, 'item'):
                                        price_value = price_value.item()
                                    
                                    # Create cluster example with CONSISTENT field names
                                    cluster_example = {
                                        'Company': example.get('Company', 'Unknown'),
                                        'TypeName': example.get('TypeName', 'Laptop'),
                                        'Title': f"{example.get('Company', 'Unknown')} {example.get('TypeName', 'Laptop')}",
                                        'Ram': f"{int(ram_val)}GB" if ram_val else "N/A",
                                        'Storage': storage,
                                        'Cpu_brand': cpu_brand,
                                        'Gpu_brand': gpu_brand,
                                        'Weight': f"{example.get('Weight', 0):.1f}kg" if example.get('Weight', 0) > 0 else "Weight N/A",
                                        'Price': f"₹{price_value:,.2f}",
                                        'Features': features_text,
                                        'Touchscreen': 'Yes' if (example.get('Touchscreen', 0) == 1 or example.get('Touchscreen') == 'Yes') else 'No',
                                        'Ips': 'Yes' if (example.get('Ips', 0) == 1 or example.get('Ips') == 'Yes') else 'No',
                                        'os': os_info,
                                        # Add alternative field names for template compatibility
                                        'Cpu': cpu_brand,
                                        'Gpu': gpu_brand,
                                        'OpSys': os_info
                                    }
                                    
                                    cluster_examples.append(cluster_example)
                                print(f"DEBUG prediction_history: Found {len(cluster_examples)} cluster examples")
                        except Exception as e:
                            print(f"ERROR prediction_history: Failed to get cluster examples: {e}")
                            cluster_examples = []
                        
                        # Save cluster category
                        print(f"DEBUG prediction_history: Saving cluster category for prediction {pid}")
                        success = save_cluster_category(
                            user_id, pid, cluster_num, cluster_name, 
                            cluster_description, cluster_examples
                        )
                        if success:
                            print(f"SUCCESS prediction_history: Cluster category saved")
                            flash(f'Category "{cluster_name}" saved successfully!', 'success')
                        else:
                            print(f"ERROR prediction_history: Failed to save cluster category")
                            flash('Could not save category information', 'warning')
                    except Exception as e:
                        print(f"ERROR prediction_history: Cluster prediction failed: {e}")
                else:
                    print("DEBUG prediction_history: KMeans model not available")

                # FIXED: Enhanced recommendations with consistent data structure
                recommendations = []
                if knn_model and hasattr(knn_model, 'get_similar_laptops'):
                    try:
                        # Ensure feature_weights is set to avoid the error
                        if not hasattr(knn_model, 'feature_weights') or knn_model.feature_weights is None:
                            n_features = X_transformed.shape[1]
                            default_weights = np.ones(n_features)
                            if hasattr(knn_model, 'set_feature_weights'):
                                knn_model.set_feature_weights(default_weights)
                                print(f"DEBUG: Set default feature weights for KNN: {default_weights.shape}")
                        
                        # Get recommendations using the same method as index page
                        recommendations = knn_model.get_similar_laptops(X_transformed, df, top_n=5, price_range_factor=0.3)
                        print(f"DEBUG: KNN recommendations found: {len(recommendations)}")
                        
                        # Ensure consistent field names for the template
                        for rec in recommendations:
                            # Map field names to be consistent with template expectations
                            rec['Cpu'] = rec.get('Cpu_brand', rec.get('Cpu', 'Unknown CPU'))
                            rec['Gpu'] = rec.get('Gpu_brand', rec.get('Gpu', 'Unknown GPU'))
                            rec['OpSys'] = rec.get('os', rec.get('OpSys', 'Unknown OS'))
                            rec['similarity_score'] = float(rec.get('Similarity', 0.5))
                            
                    except Exception as e:
                        print(f"WARNING: Enhanced KNN failed: {e}")
                        recommendations = []

                # FIXED: Fallback with consistent data structure
                if not recommendations:
                    print("DEBUG: Using fallback recommendation logic with consistent structure")
                    try:
                        X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                        if hasattr(X_train_transformed, 'toarray'):
                            X_train_transformed = X_train_transformed.toarray()
                        
                        similarities = []
                        query_vector = X_transformed[0]
                        
                        for i, train_sample in enumerate(X_train_transformed):
                            # Simple cosine similarity
                            dot_product = np.dot(query_vector, train_sample)
                            norm_query = np.linalg.norm(query_vector)
                            norm_train = np.linalg.norm(train_sample)
                            
                            if norm_query > 0 and norm_train > 0:
                                similarity = dot_product / (norm_query * norm_train)
                            else:
                                similarity = 0.5
                                
                            # Price-based filtering
                            laptop_price = df.iloc[i]['Price']
                            if hasattr(laptop_price, 'item'):
                                laptop_price = laptop_price.item()
                                
                            # Boost similarity if price is in reasonable range
                            if 0.7 * predicted_price <= laptop_price <= 1.3 * predicted_price:
                                similarity *= 1.2
                                
                            similarities.append((similarity, i))
                        
                        # Sort by similarity and get top matches
                        similarities.sort(reverse=True)
                        
                        seen_companies = set()
                        for similarity_score, idx in similarities:
                            if len(recommendations) >= 5:
                                break
                                
                            rec = df.iloc[idx].to_dict()
                            company = rec.get('Company', 'Unknown')
                            
                            # Ensure diversity - don't show too many from same company
                            if len([r for r in recommendations if r.get('Company') == company]) >= 2:
                                continue
                                
                            seen_companies.add(company)
                            
                            # Extract laptop details with CONSISTENT field names
                            ssd_val = rec.get('SSD', 0) or 0
                            hdd_val = rec.get('HDD', 0) or 0
                            ram_val = rec.get('Ram', 0) or 0
                            
                            # Skip if no storage (invalid laptop)
                            if ssd_val == 0 and hdd_val == 0:
                                continue
                            
                            # Build storage description
                            storage_parts = []
                            if ssd_val > 0:
                                storage_parts.append(f"{int(ssd_val)}GB SSD")
                            if hdd_val > 0:
                                storage_parts.append(f"{int(hdd_val)}GB HDD")
                            storage = " + ".join(storage_parts) if storage_parts else "Unknown storage"
                            
                            # Get other specs with CONSISTENT field names
                            cpu_brand = rec.get('Cpu brand', rec.get('Cpu_brand', 'Unknown CPU'))
                            gpu_brand = rec.get('Gpu brand', rec.get('Gpu_brand', 'Unknown GPU'))
                            os_info = rec.get('os', rec.get('OpSys', 'Unknown OS'))
                            weight_val = rec.get('Weight', 0) or 0
                            price_val = rec.get('Price', 0) or 0
                            
                            if hasattr(price_val, 'item'):
                                price_val = price_val.item()
                            
                            # Build features list
                            features = []
                            if rec.get('Touchscreen', 0) == 1:
                                features.append('Touchscreen')
                            if rec.get('Ips', 0) == 1:
                                features.append('IPS Display')
                            if ram_val >= 16:
                                features.append('High Memory')
                            elif ram_val >= 8:
                                features.append('Good Memory')
                                
                            features_text = ', '.join(features) if features else 'Standard Features'
                            
                            # Create recommendation with CONSISTENT field names matching the template
                            recommendation = {
                                'Title': f"{company} {rec.get('TypeName', 'Laptop')}",
                                'Company': company,
                                'TypeName': rec.get('TypeName', 'Laptop'),
                                'Ram': f"{int(ram_val)}GB",
                                'Storage': storage,
                                'Cpu': cpu_brand,  # Use 'Cpu' instead of 'Cpu_brand'
                                'Gpu': gpu_brand,  # Use 'Gpu' instead of 'Gpu_brand'
                                'Weight': f"{weight_val:.1f}kg" if weight_val > 0 else "Weight N/A",
                                'Price': f"₹{price_val:,.2f}",
                                'similarity_score': float(similarity_score),
                                'Features': features_text,
                                'Touchscreen': 'Yes' if rec.get('Touchscreen', 0) == 1 else 'No',
                                'Ips': 'Yes' if rec.get('Ips', 0) == 1 else 'No',
                                'OpSys': os_info,  # Use 'OpSys' consistently
                                'Cpu_brand': cpu_brand,  # Keep both for compatibility
                                'Gpu_brand': gpu_brand,  # Keep both for compatibility
                                'os': os_info  # Keep both for compatibility
                            }
                            
                            recommendations.append(recommendation)
                            
                        print(f"DEBUG: Fallback recommendations generated: {len(recommendations)}")
                        
                    except Exception as fallback_error:
                        print(f"ERROR: Fallback recommendation also failed: {fallback_error}")
                        recommendations = []

                # FIXED: Save recommendations to database
                if recommendations and pid:
                    print(f"DEBUG: Starting to save {len(recommendations)} recommendations for PID: {pid}")
                    rec_connection = create_connection()
                    if rec_connection:
                        try:
                            with rec_connection.cursor() as cursor_rec:
                                saved_count = 0
                                for rec in recommendations:
                                    try:
                                        # Build comprehensive specs - FIXED FIELD NAMES
                                        specs_parts = []
                                        if rec.get('Ram'):
                                            specs_parts.append(f"RAM: {rec['Ram']}")
                                        if rec.get('Storage'):
                                            specs_parts.append(f"Storage: {rec['Storage']}")
                                        if rec.get('Cpu'):
                                            specs_parts.append(f"CPU: {rec['Cpu']}")
                                        elif rec.get('Cpu_brand'):  # Handle alternative field name
                                            specs_parts.append(f"CPU: {rec['Cpu_brand']}")
                                        if rec.get('Gpu'):
                                            specs_parts.append(f"GPU: {rec['Gpu']}")
                                        elif rec.get('Gpu_brand'):  # Handle alternative field name
                                            specs_parts.append(f"GPU: {rec['Gpu_brand']}")
                                        
                                        specs = ', '.join(specs_parts) if specs_parts else "Standard specifications"
                                        
                                        # Ensure price is numeric - FIXED PRICE HANDLING
                                        price_val = rec.get('Price', 0)
                                        if isinstance(price_val, str):
                                            if '₹' in price_val:
                                                try:
                                                    price_val = float(price_val.replace('₹', '').replace(',', ''))
                                                except (ValueError, TypeError):
                                                    price_val = 0
                                            else:
                                                try:
                                                    price_val = float(price_val)
                                                except (ValueError, TypeError):
                                                    price_val = 0
                                        
                                        # Ensure similarity score is numeric
                                        similarity_val = rec.get('similarity_score', 0.5)
                                        if hasattr(similarity_val, 'item'):
                                            similarity_val = similarity_val.item()
                                        similarity_val = float(similarity_val)
                                        
                                        # Get laptop name - FIXED TITLE FIELD
                                        laptop_name = rec.get('Title', f"{rec.get('Company', 'Unknown')} {rec.get('TypeName', 'Laptop')}")
                                        
                                        print(f"DEBUG: Saving recommendation - Name: {laptop_name}, Price: {price_val}, Similarity: {similarity_val}")
                                        
                                        # Insert into database
                                        cursor_rec.execute(
                                            """
                                            INSERT INTO recommendations (uid, pid, laptop_name, specs, price, similarity_score)
                                            VALUES (%s, %s, %s, %s, %s, %s)
                                            """,
                                            (user_id, pid, laptop_name, specs, price_val, similarity_val)
                                        )
                                        saved_count += 1
                                        print(f"DEBUG: Successfully saved recommendation {saved_count}: {laptop_name}")
                                        
                                    except Exception as rec_error:
                                        print(f"ERROR: Failed to save individual recommendation: {rec_error}")
                                        import traceback
                                        traceback.print_exc()
                                        continue
                                
                                rec_connection.commit()
                                print(f"SUCCESS: Saved {saved_count}/{len(recommendations)} recommendations for PID: {pid}")
                                if saved_count > 0:
                                    flash(f'Saved {saved_count} recommendations to your history!', 'success')
                                else:
                                    flash('No recommendations could be saved due to errors', 'warning')
                                
                        except Exception as e:
                            print(f"ERROR: Failed to save recommendations to database: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            flash(f"Could not save recommendations: {str(e)}", 'warning')
                        finally:
                            close_connection(rec_connection)
                    else:
                        print("ERROR: Could not create database connection for saving recommendations")
                        flash("Database connection failed - recommendations not saved", 'warning')
                else:
                    print(f"DEBUG: Cannot save recommendations - PID: {pid}, Recommendations count: {len(recommendations) if recommendations else 0}")
                    if not pid:
                        flash("Warning: Could not save recommendations due to missing prediction ID", 'warning')
                    elif not recommendations:
                        flash("No recommendations generated to save", 'info')

                # DEBUG: Check what we're about to return
                print(f"DEBUG FINAL: Prediction completed for user {user_id}")
                print(f"  - Predicted price: {formatted_price}")
                print(f"  - Recommendations count: {len(recommendations)}")
                print(f"  - PID for saving: {pid}")
                print(f"  - Cluster label: {cluster_label}")
                print(f"  - Cluster examples: {len(cluster_examples)}")

                return render_template('prediction.html',
                                      predicted_price=formatted_price,
                                      recommendations=recommendations,
                                      username=username,
                                      form_data=form_data,
                                      cluster_label=cluster_label,
                                      cluster_examples=cluster_examples)

            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                print(f"ERROR prediction_history: {error_msg}")
                import traceback
                traceback.print_exc()
                flash(error_msg, 'error')
                return redirect(url_for('prediction_history'))

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
        
        # Fetch prediction details
        cursor.execute("""
            SELECT * FROM predictions WHERE pid = %s AND uid = %s
        """, (pid, user_id))
        prediction = cursor.fetchone()
        
        if not prediction:
            flash('Prediction not found or you do not have permission to view it.', 'error')
            return redirect(url_for('prediction_history'))

        # Fetch category information from cluster_categories table
        cursor.execute("""
            SELECT cluster_name, cluster_description, example_laptops 
            FROM cluster_categories 
            WHERE pid = %s AND uid = %s
        """, (pid, user_id))
        category_data = cursor.fetchone()

        # Prepare form data for display
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

        # Fetch recommendations
        cursor.execute("""
            SELECT laptop_name, specs, price, similarity_score
            FROM recommendations
            WHERE uid = %s AND pid = %s
            ORDER BY saved_at DESC
            LIMIT 5
        """, (user_id, pid))
        recommendations_raw = cursor.fetchall()

        # Process recommendations
        recommendations = []
        for rec in recommendations_raw:
            try:
                specs = rec['specs'] or ""
                company = rec['laptop_name'].split()[0] if rec['laptop_name'] and len(rec['laptop_name'].split()) > 0 else "Unknown"
                
                # Parse specs
                ram = "N/A"
                cpu = "Unknown"
                gpu = "Unknown"
                storage = "N/A"

                if "RAM:" in specs:
                    ram = specs.split("RAM:")[1].split(",")[0].strip()
                if "CPU:" in specs:
                    cpu = specs.split("CPU:")[1].split(",")[0].strip()
                if "GPU:" in specs:
                    gpu = specs.split("GPU:")[1].split(",")[0].strip()
                if "Storage:" in specs:
                    storage = specs.split("Storage:")[1].split(",")[0].strip()

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
                    'Features': "Standard Features",
                    'Touchscreen': 'No',  # Default values
                    'Ips': 'No'
                })
            except Exception as e:
                print(f"Error parsing recommendation: {e}")
                # Fallback recommendation format
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
                    'Features': "Standard Features",
                    'Touchscreen': 'No',
                    'Ips': 'No'
                })

        # Process category data
        cluster_label = None
        cluster_examples = []
        
        if category_data:
            cluster_label = category_data['cluster_name']
            
            # Parse example laptops from JSON
            if category_data['example_laptops']:
                try:
                    cluster_examples = json.loads(category_data['example_laptops'])
                    print(f"DEBUG: Loaded {len(cluster_examples)} cluster examples")
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse cluster examples JSON: {e}")
                    cluster_examples = []
        else:
            print(f"DEBUG: No category data found for prediction {pid}")

        # Get username
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        username = user['username'] if user else 'User'

        print(f"DEBUG: Rendering prediction page with - Cluster: {cluster_label}, Examples: {len(cluster_examples)}")

        return render_template('prediction.html',
                              predicted_price=f"₹{prediction['predicted_price']:,.2f}",
                              recommendations=recommendations,
                              username=username,
                              form_data=form_data,
                              cluster_label=cluster_label,
                              cluster_examples=cluster_examples)

    except Exception as e:
        flash(f"Error fetching prediction details: {str(e)}", 'error')
        print(f"ERROR in view_prediction: {str(e)}")
        import traceback
        traceback.print_exc()
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

# =============================================================================
# ADMIN DASHBOARD ROUTES
# =============================================================================

@app.route('/admindashboard')
def admindashboard():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    stats = {
        'total_users': 0,
        'total_predictions': 0,
        'average_price': 0,
        'total_bookings': 0
    }
    recent_users = []
    recent_predictions = []

    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM users")
                stats['total_users'] = cursor.fetchone()['count']

                cursor.execute("SELECT COUNT(*) as count FROM predictions")
                stats['total_predictions'] = cursor.fetchone()['count']

                cursor.execute("SELECT AVG(predicted_price) as avg_price FROM predictions")
                avg_result = cursor.fetchone()
                stats['average_price'] = round(avg_result['avg_price'], 2) if avg_result and avg_result['avg_price'] else 0

                cursor.execute("SELECT COUNT(*) as count FROM bookings")
                stats['total_bookings'] = cursor.fetchone()['count']

                cursor.execute("SELECT uid, username, email, created_at FROM users ORDER BY created_at DESC LIMIT 5")
                recent_users = cursor.fetchall()

                cursor.execute("""
                    SELECT p.pid, u.username, p.predicted_price, p.created_at, p.company, p.type, p.ram
                    FROM predictions p 
                    JOIN users u ON p.uid = u.uid 
                    ORDER BY p.created_at DESC 
                    LIMIT 5
                """)
                recent_predictions = cursor.fetchall()

        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('admindashboard.html', 
                         stats=stats, 
                         recent_users=recent_users, 
                         recent_predictions=recent_predictions)

@app.route('/user_list')
def user_list():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    users = []
    
    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT uid, username, email, created_at FROM users ORDER BY created_at DESC")
                users = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('user_list.html', users=users)

@app.route('/all_predictions')
def all_predictions():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    predictions = []
    
    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("""
                    SELECT p.*, u.username 
                    FROM predictions p 
                    JOIN users u ON p.uid = u.uid 
                    ORDER BY p.created_at DESC
                """)
                predictions = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('all_predictions.html', predictions=predictions)

@app.route('/get_prediction_details/<int:pid>')
def get_prediction_details(pid):
    """AJAX endpoint to get prediction details for admin view"""
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'error': 'Admin authentication required'}), 401

    connection = create_connection()
    if not connection:
        return jsonify({'success': False, 'error': 'Database connection error'})

    try:
        with connection.cursor(dictionary=True) as cursor:
            # Get detailed prediction information with user details
            cursor.execute("""
                SELECT 
                    p.pid,
                    p.uid,
                    p.company,
                    p.type,
                    p.ram,
                    p.weight,
                    p.touchscreen,
                    p.ips,
                    p.screen_size,
                    p.resolution,
                    p.cpu,
                    p.hdd,
                    p.ssd,
                    p.gpu,
                    p.os,
                    p.predicted_price,
                    p.created_at,
                    u.username
                FROM predictions p 
                JOIN users u ON p.uid = u.uid 
                WHERE p.pid = %s
            """, (pid,))
            
            prediction = cursor.fetchone()
            
            if not prediction:
                return jsonify({'success': False, 'error': 'Prediction not found'})

            # Format the prediction data for JSON response
            prediction_data = {
                'pid': prediction['pid'],
                'username': prediction['username'],
                'company': prediction['company'],
                'type': prediction['type'],
                'ram': prediction['ram'],
                'weight': prediction['weight'],
                'touchscreen': bool(prediction['touchscreen']),
                'ips': bool(prediction['ips']),
                'screen_size': prediction['screen_size'],
                'resolution': prediction['resolution'],
                'cpu': prediction['cpu'],
                'hdd': prediction['hdd'],
                'ssd': prediction['ssd'],
                'gpu': prediction['gpu'],
                'os': prediction['os'],
                'predicted_price': float(prediction['predicted_price']) if prediction['predicted_price'] else 0,
                'created_at': prediction['created_at'].strftime('%Y-%m-%d %H:%M') if prediction['created_at'] else 'N/A'
            }

            return jsonify({
                'success': True,
                'prediction': prediction_data
            })

    except Exception as e:
        print(f"Error fetching prediction details: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})
    
    finally:
        close_connection(connection)

@app.route('/all_recommendations')
def all_recommendations():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if not connection:
        flash('Database connection error.', 'error')
        return render_template('all_recommendations.html', 
                             prediction_groups=[],
                             total_recommendations=0,
                             total_users=0,
                             average_match=0,
                             best_price=0)

    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get recommendations grouped by prediction with prediction details and username
        cursor.execute("""
            SELECT 
                r.rid, 
                r.laptop_name, 
                r.specs, 
                r.price, 
                r.similarity_score, 
                r.saved_at,
                r.pid,
                r.uid,
                u.username,
                p.pid as prediction_id,
                p.created_at as prediction_date,
                p.company,
                p.type,
                p.ram,
                p.cpu,
                p.predicted_price,
                CONCAT(COALESCE(p.company, 'Unknown'), ' ', 
                       COALESCE(p.type, 'Laptop'), ' - ', 
                       COALESCE(p.ram, 0), 'GB RAM, ', 
                       COALESCE(p.cpu, 'Unknown CPU')) as search_criteria
            FROM recommendations r
            JOIN users u ON r.uid = u.uid
            LEFT JOIN predictions p ON r.pid = p.pid
            ORDER BY p.created_at DESC, r.saved_at DESC
        """)
        
        recommendations_raw = cursor.fetchall()

        # Group recommendations by prediction
        prediction_groups = {}
        total_recommendations = 0
        all_similarities = []
        all_prices = []
        unique_users = set()

        for rec in recommendations_raw:
            prediction_id = rec['pid'] or 'unknown_' + str(rec['rid'])
            
            if prediction_id not in prediction_groups:
                # Create prediction group
                prediction_groups[prediction_id] = {
                    'prediction_id': rec['prediction_id'],
                    'username': rec['username'],
                    'search_criteria': rec['search_criteria'] or 'Custom Search',
                    'prediction_date': rec['prediction_date'],
                    'predicted_price': rec['predicted_price'],
                    'recommendations': []
                }
            
            # Add recommendation to group
            prediction_groups[prediction_id]['recommendations'].append({
                'rid': rec['rid'],
                'laptop_name': rec['laptop_name'],
                'specs': rec['specs'],
                'price': rec['price'],
                'similarity_score': rec['similarity_score'],
                'saved_at': rec['saved_at']
            })
            
            total_recommendations += 1
            unique_users.add(rec['uid'])
            
            if rec['similarity_score']:
                all_similarities.append(float(rec['similarity_score']))
            if rec['price']:
                all_prices.append(float(rec['price']))

        # Convert to list for template
        prediction_groups_list = list(prediction_groups.values())
        
        # Calculate stats
        average_match = 0
        best_price = 0
        total_users = len(unique_users)
        
        if all_similarities:
            average_match = round((sum(all_similarities) / len(all_similarities)) * 100)
        if all_prices:
            best_price = min(all_prices)

        print(f"DEBUG: Found {total_recommendations} recommendations in {len(prediction_groups_list)} prediction groups from {total_users} users")

        return render_template('all_recommendations.html', 
                             prediction_groups=prediction_groups_list,
                             total_recommendations=total_recommendations,
                             total_users=total_users,
                             average_match=average_match,
                             best_price=best_price)

    except Exception as e:
        flash(f"Error fetching recommendations: {str(e)}", 'error')
        import traceback
        traceback.print_exc()
        return render_template('all_recommendations.html', 
                             prediction_groups=[],
                             total_recommendations=0,
                             total_users=0,
                             average_match=0,
                             best_price=0)
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/all_bookings')
def all_bookings():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    bookings = []
    stats = {
        'confirmed_count': 0,
        'pending_count': 0,
        'cancelled_count': 0,
        'completed_count': 0,
        'total_value': 0
    }
    
    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("""
                    SELECT b.*, u.username 
                    FROM bookings b 
                    JOIN users u ON b.uid = u.uid 
                    ORDER BY b.booked_at DESC
                """)
                bookings = cursor.fetchall()

                # Count statuses and calculate total value
                for booking in bookings:
                    status = booking.get('booking_status', 'pending')
                    if status == 'confirmed':
                        stats['confirmed_count'] += 1
                    elif status == 'pending':
                        stats['pending_count'] += 1
                    elif status == 'cancelled':
                        stats['cancelled_count'] += 1
                    elif status == 'completed':
                        stats['completed_count'] += 1
                    
                    # Calculate total value (sum of all booking prices)
                    price = booking.get('price', 0)
                    if price is not None:
                        try:
                            price_float = float(price)
                            stats['total_value'] += price_float
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid price value '{price}' for booking {booking.get('bid')}")

        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('all_bookings.html', 
                         bookings=bookings,
                         confirmed_count=stats['confirmed_count'],
                         pending_count=stats['pending_count'],
                         cancelled_count=stats['cancelled_count'],
                         completed_count=stats['completed_count'],
                         total_value=stats['total_value'])

@app.route('/all_categories')
def all_categories():
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    categories = []
    stats = {
        'total_categories': 0,
        'unique_clusters': set(),
        'total_examples': 0,
        'active_users': set()
    }
    
    if not connection:
        flash('Database connection error.', 'error')
        return render_template('all_categories.html', 
                             categories=[],
                             total_categories=0,
                             unique_clusters_count=0,
                             total_examples=0,
                             active_users_count=0)

    try:
        with connection.cursor(dictionary=True) as cursor:
            # Get all categories with user information
            cursor.execute("""
                SELECT 
                    cc.cid, 
                    cc.uid,
                    cc.cluster_number, 
                    cc.cluster_name, 
                    cc.cluster_description, 
                    cc.example_laptops, 
                    cc.created_at,
                    cc.pid,
                    u.username,
                    p.predicted_price,
                    p.company,
                    p.type,
                    p.ram,
                    CONCAT(COALESCE(p.company, 'Unknown'), ' ', 
                           COALESCE(p.type, 'Laptop'), ' - ', 
                           COALESCE(p.ram, 0), 'GB RAM') as prediction_details
                FROM cluster_categories cc 
                JOIN users u ON cc.uid = u.uid 
                LEFT JOIN predictions p ON cc.pid = p.pid
                ORDER BY cc.created_at DESC
            """)
            categories_raw = cursor.fetchall()

            # Process categories and parse JSON
            for cat in categories_raw:
                try:
                    # Parse example laptops from JSON
                    example_laptops = []
                    if cat['example_laptops']:
                        try:
                            examples_data = json.loads(cat['example_laptops'])
                            # Ensure we have a list and count the examples
                            if isinstance(examples_data, list):
                                example_laptops = examples_data
                                stats['total_examples'] += len(examples_data)
                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"Error parsing category {cat['cid']} examples: {e}")
                            example_laptops = []

                    # Add user to active users count
                    stats['active_users'].add(cat['uid'])
                    
                    # Add cluster to unique clusters
                    stats['unique_clusters'].add(cat['cluster_number'])

                    # Build category data
                    category_data = {
                        'cid': cat['cid'],
                        'uid': cat['uid'],
                        'cluster_number': cat['cluster_number'],
                        'cluster_name': cat['cluster_name'],
                        'cluster_description': cat['cluster_description'],
                        'example_laptops': example_laptops,
                        'created_at': cat['created_at'],
                        'username': cat['username'],
                        'predicted_price': cat['predicted_price'],
                        'prediction_details': cat['prediction_details']
                    }
                    
                    categories.append(category_data)
                    
                except Exception as e:
                    print(f"Error processing category {cat.get('cid', 'unknown')}: {e}")
                    # Add basic category data even if examples fail
                    categories.append({
                        'cid': cat['cid'],
                        'uid': cat['uid'],
                        'cluster_number': cat['cluster_number'],
                        'cluster_name': cat['cluster_name'],
                        'cluster_description': cat['cluster_description'],
                        'example_laptops': [],
                        'created_at': cat['created_at'],
                        'username': cat['username'],
                        'predicted_price': cat['predicted_price'],
                        'prediction_details': cat['prediction_details']
                    })
                    stats['unique_clusters'].add(cat['cluster_number'])
                    stats['active_users'].add(cat['uid'])

            # Calculate final stats - convert sets to counts
            stats['total_categories'] = len(categories)
            stats['unique_clusters_count'] = len(stats['unique_clusters'])
            stats['active_users_count'] = len(stats['active_users'])

            print(f"DEBUG: Found {stats['total_categories']} categories with {stats['total_examples']} total examples")

    except Exception as e:
        flash(f"An error occurred while fetching categories: {e}", 'error')
        print(f"ERROR in all_categories: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_connection(connection)

    return render_template('all_categories.html', 
                         categories=categories,
                         total_categories=stats['total_categories'],
                         unique_clusters=stats['unique_clusters_count'],  # Pass the count, not the set
                         total_examples=stats['total_examples'],
                         active_users=stats['active_users_count'])  # Pass the count, not the set

# Admin Delete Routes
@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM bookings WHERE uid = %s", (user_id,))
                cursor.execute("DELETE FROM recommendations WHERE uid = %s", (user_id,))
                cursor.execute("DELETE FROM cluster_categories WHERE uid = %s", (user_id,))
                cursor.execute("DELETE FROM predictions WHERE uid = %s", (user_id,))
                cursor.execute("DELETE FROM users WHERE uid = %s", (user_id,))
                connection.commit()
                flash('User and all associated data deleted successfully!', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('user_list'))

@app.route('/delete_prediction_admin/<int:pid>')
def delete_prediction_admin(pid):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM predictions WHERE pid = %s", (pid,))
                connection.commit()
                flash('Prediction deleted successfully!', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('all_predictions'))

@app.route('/delete_recommendation_admin/<int:rid>')
def delete_recommendation_admin(rid):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM recommendations WHERE rid = %s", (rid,))
                connection.commit()
                flash('Recommendation deleted successfully!', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('all_recommendations'))

@app.route('/delete_booking_admin/<int:bid>')
def delete_booking_admin(bid):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM bookings WHERE bid = %s", (bid,))
                connection.commit()
                flash('Booking deleted successfully!', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('all_bookings'))

@app.route('/delete_category_admin/<int:cid>')
def delete_category_admin(cid):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM cluster_categories WHERE cid = %s", (cid,))
                connection.commit()
                flash('Category deleted successfully!', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('all_categories'))

@app.route('/edit_user/<int:user_id>')
def edit_user(user_id):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    user = None
    
    if connection:
        try:
            with connection.cursor(dictionary=True) as cursor:
                cursor.execute("SELECT uid, username, email, created_at FROM users WHERE uid = %s", (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    flash('User not found!', 'error')
                    return redirect(url_for('user_list'))
                    
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
            return redirect(url_for('user_list'))
        finally:
            close_connection(connection)

    # Pass an empty errors dictionary to avoid the template error
    return render_template('edit_user.html', user=user, errors={})

@app.route('/update_user/<int:user_id>', methods=['POST'])
def update_user(user_id):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    # Get form data
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirmPassword')

    # Validation
    errors = {}
    
    # Username validation
    if not username or len(username) < 3:
        errors['username'] = 'Username must be at least 3 characters long'
    elif not re.match(r'^[A-Za-z][A-Za-z0-9_]{2,19}$', username):
        errors['username'] = 'Username must start with a letter and contain only letters, numbers, and underscores'
    
    # Email validation
    if not email or not re.match(r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$', email):
        errors['email'] = 'Please enter a valid email address'
    
    # Password validation (only if provided)
    if password:
        if len(password) < 8:
            errors['password'] = 'Password must be at least 8 characters long'
        elif not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', password):
            errors['password'] = 'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
        elif password != confirm_password:
            errors['confirmPassword'] = 'Passwords do not match'

    # Backend validation for duplicate username and email
    connection = create_connection()
    if not connection:
        flash('Database connection error', 'error')
        return redirect(url_for('user_list'))

    try:
        with connection.cursor() as cursor:
            # Check if username already exists (excluding current user)
            cursor.execute(
                "SELECT uid, username FROM users WHERE username = %s AND uid != %s",
                (username, user_id)
            )
            existing_username = cursor.fetchone()
            
            # Check if email already exists (excluding current user)
            cursor.execute(
                "SELECT uid, email FROM users WHERE email = %s AND uid != %s",
                (email, user_id)
            )
            existing_email = cursor.fetchone()
            
            if existing_username:
                errors['username'] = 'Username already exists! Please choose a different username.'
            
            if existing_email:
                errors['email'] = 'Email already exists! Please use a different email address.'
                
    except Exception as e:
        flash(f'Database error during validation: {str(e)}', 'error')
        return redirect(url_for('user_list'))
    finally:
        close_connection(connection)

    # If there are validation errors, return to form with errors
    if errors:
        # Re-fetch user data to display form with errors
        connection = create_connection()
        user = None
        if connection:
            try:
                with connection.cursor(dictionary=True) as cursor:
                    cursor.execute("SELECT uid, username, email, created_at FROM users WHERE uid = %s", (user_id,))
                    user = cursor.fetchone()
            except Exception as e:
                flash(f"An error occurred: {e}", 'error')
            finally:
                close_connection(connection)
        
        return render_template('edit_user.html', user=user, errors=errors)

    # If no errors, proceed with update
    connection = create_connection()
    if not connection:
        flash('Database connection error', 'error')
        return redirect(url_for('user_list'))

    try:
        with connection.cursor() as cursor:
            # Update user
            if password:
                # Update with new password
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                cursor.execute(
                    "UPDATE users SET username = %s, email = %s, password = %s WHERE uid = %s",
                    (username, email, hashed_password, user_id)
                )
            else:
                # Update without changing password
                cursor.execute(
                    "UPDATE users SET username = %s, email = %s WHERE uid = %s",
                    (username, email, user_id)
                )
            
            connection.commit()
            flash('User updated successfully!', 'success')
            
    except Exception as e:
        flash(f'Error updating user: {str(e)}', 'error')
        connection.rollback()
    finally:
        close_connection(connection)

    return redirect(url_for('user_list'))

@app.route('/update_booking_status/<int:bid>', methods=['POST'])
def update_booking_status(bid):
    if 'admin_logged_in' not in session:
        flash('Please login as admin', 'warning')
        return redirect(url_for('admin_login'))

    new_status = request.form.get('status')
    valid_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
    
    if new_status not in valid_statuses:
        flash('Invalid status', 'error')
        return redirect(url_for('all_bookings'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE bookings 
                    SET booking_status = %s, updated_at = NOW() 
                    WHERE bid = %s
                """, (new_status, bid))
                connection.commit()
                flash(f'Booking status updated to {new_status.title()}!', 'success')
        except Exception as e:
            flash(f'Error updating booking status: {str(e)}', 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('all_bookings'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('admin_logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)