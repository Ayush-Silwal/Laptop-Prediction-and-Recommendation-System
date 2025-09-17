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

        if df.empty or preprocessor is None or rf_model is None or knn_model is None or kmeans_model is None:
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
        kmeans_model = None

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
        formatted_price = f"RS {predicted_price:,.2f}"  # Fixed: Single RS prefix

        # Recommendations using CustomKNN
        recommendations = []
        if knn_model and hasattr(knn_model, 'get_similar_laptops'):
            recommendations = knn_model.get_similar_laptops(X_transformed, df, top_n=5)
        else:
            # Fallback: Manual KNN recommendation with improved formatting
            distances = []
            X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
            for i, sample in enumerate(X_train_transformed):
                cosine_sim = knn_model._cosine_similarity(X_transformed[0], sample)
                distances.append((cosine_sim, i))
            top_indices = [idx for _, idx in sorted(distances, reverse=True)[:5]]
            for i, idx in enumerate(top_indices):
                rec = df.iloc[idx].to_dict()
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
                features_text = ', '.join(features) if features else 'Standard Features'
                recommendations.append({
                    'Company': rec.get('Company', 'Unknown'),
                    'TypeName': rec.get('TypeName', 'Laptop'),
                    'Title': f"{rec.get('Company', 'Unknown')} {rec.get('TypeName', 'Laptop')}",
                    'Ram': f"{int(rec.get('Ram', 0))}GB",
                    'Storage': storage,
                    'Cpu_brand': rec.get('Cpu brand', 'Unknown'),
                    'Gpu_brand': rec.get('Gpu brand', 'Unknown'),
                    'Weight': f"{rec.get('Weight', 0):.1f}kg" if rec.get('Weight', 0) > 0 else "Weight N/A",
                    'Price': f"RS {rec.get('Price', 0):,.2f}",
                    'Similarity': f"{distances[i][0]:.2f}",
                    'Features': features_text,
                    'Touchscreen': 'Yes' if rec.get('Touchscreen', 0) else 'No',
                    'Ips': 'Yes' if rec.get('Ips', 0) else 'No',
                    'os': rec.get('os', 'Unknown OS')
                })

        # Clustering using CustomKMeans
        cluster_label = None
        cluster_examples = []
        cluster_name = "Unknown Cluster"
        if kmeans_model:
            cluster_label = kmeans_model.predict(X_transformed)[0]
            # Use cluster_names if available, else fallback to DEFAULT_CLUSTER_NAMES
            cluster_name = getattr(kmeans_model, 'cluster_names', DEFAULT_CLUSTER_NAMES).get(cluster_label, f"Cluster {cluster_label}")
            if hasattr(kmeans_model, 'get_cluster_examples'):
                X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                if hasattr(X_train_transformed, 'toarray'):
                    X_train_transformed = X_train_transformed.toarray()
                cluster_examples = kmeans_model.get_cluster_examples(cluster_label, df, X_all=X_train_transformed, top_n=5)
            else:
                # Fallback: Select top 5 laptops from the same cluster with improved formatting
                X_train_transformed = preprocessor.transform(df.drop(columns=['Price']))
                if hasattr(X_train_transformed, 'toarray'):
                    X_train_transformed = X_train_transformed.toarray()
                cluster_labels = kmeans_model.predict(X_train_transformed)
                cluster_indices = np.where(cluster_labels == cluster_label)[0][:5]
                for idx in cluster_indices:
                    example = df.iloc[idx].to_dict()
                    storage_parts = []
                    if example.get('SSD', 0) > 0:
                        storage_parts.append(f"{int(example['SSD'])}GB SSD")
                    if example.get('HDD', 0) > 0:
                        storage_parts.append(f"{int(example['HDD'])}GB HDD")
                    storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                    features = []
                    if example.get('Touchscreen', 0):
                        features.append('Touchscreen')
                    if example.get('Ips', 0):
                        features.append('IPS Display')
                    features_text = ', '.join(features) if features else 'Standard Features'
                    cluster_examples.append({
                        'Company': example.get('Company', 'Unknown'),
                        'TypeName': example.get('TypeName', 'Laptop'),
                        'Title': f"{example.get('Company', 'Unknown')} {example.get('TypeName', 'Laptop')}",
                        'Ram': f"{int(example.get('Ram', 0))}GB",
                        'Storage': storage,
                        'Cpu_brand': example.get('Cpu brand', 'Unknown'),
                        'Gpu_brand': example.get('Gpu brand', 'Unknown'),
                        'Weight': f"{example.get('Weight', 0):.1f}kg" if example.get('Weight', 0) > 0 else "Weight N/A",
                        'Price': f"RS {example.get('Price', 0):,.2f}",
                        'Features': features_text,
                        'Touchscreen': 'Yes' if example.get('Touchscreen', 0) else 'No',
                        'Ips': 'Yes' if example.get('Ips', 0) else 'No',
                        'os': example.get('os', 'Unknown OS')
                    })

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
                except Exception as e:
                    print(f"Error saving prediction to database: {str(e)}")
                finally:
                    close_connection(connection)

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

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_id' not in session:
        flash('Please login to get recommendations', 'warning')
        return redirect(url_for('login'))

    try:
        # Get user preferences from form
        budget = float(request.form.get('budget', 50000))
        min_ram = int(request.form.get('min_ram', 8))
        min_ssd = int(request.form.get('min_ssd', 256))
        preferred_brands = request.form.getlist('preferred_brands')
        preferred_cpu = request.form.get('preferred_cpu', '')

        # Filter the dataset based on preferences
        filtered_df = df.copy()
        filtered_df = filtered_df[filtered_df['Price'] <= budget]
        filtered_df = filtered_df[filtered_df['Ram'] >= min_ram]
        filtered_df = filtered_df[filtered_df['SSD'] >= min_ssd]

        if preferred_brands:
            filtered_df = filtered_df[filtered_df['Company'].isin(preferred_brands)]
        if preferred_cpu:
            filtered_df = filtered_df[filtered_df['Cpu brand'] == preferred_cpu]

        # If no results, relax filters
        if len(filtered_df) == 0:
            flash('No exact matches found. Showing closest options.', 'info')
            filtered_df = df.copy()
            filtered_df = filtered_df[filtered_df['Price'] <= budget * 1.2]
            filtered_df = filtered_df[filtered_df['Ram'] >= max(4, min_ram - 4)]

        # Use KMeans clustering for recommendations
        recommendations = []
        if kmeans_model and len(filtered_df) > 5:
            try:
                X_filtered = preprocessor.transform(filtered_df.drop(columns=['Price']))
                if hasattr(X_filtered, 'toarray'):
                    X_filtered = X_filtered.toarray()
                clusters = kmeans_model.predict(X_filtered)
                filtered_df['cluster'] = clusters

                # Get top laptops from each cluster
                for cluster_id in np.unique(clusters):
                    if hasattr(kmeans_model, 'get_cluster_examples'):
                        cluster_examples = kmeans_model.get_cluster_examples(cluster_id, filtered_df, X_filtered, top_n=1)
                    else:
                        # Fallback: Select top laptop from cluster with improved formatting
                        cluster_indices = filtered_df[filtered_df['cluster'] == cluster_id].index[:1]
                        cluster_examples = []
                        for idx in cluster_indices:
                            example = filtered_df.loc[idx].to_dict()
                            storage_parts = []
                            if example.get('SSD', 0) > 0:
                                storage_parts.append(f"{int(example['SSD'])}GB SSD")
                            if example.get('HDD', 0) > 0:
                                storage_parts.append(f"{int(example['HDD'])}GB HDD")
                            storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                            features = []
                            if example.get('Touchscreen', 0):
                                features.append('Touchscreen')
                            if example.get('Ips', 0):
                                features.append('IPS Display')
                            features_text = ', '.join(features) if features else 'Standard Features'
                            cluster_examples.append({
                                'Company': example.get('Company', 'Unknown'),
                                'TypeName': example.get('TypeName', 'Laptop'),
                                'Title': f"{example.get('Company', 'Unknown')} {example.get('TypeName', 'Laptop')}",
                                'Ram': f"{int(example.get('Ram', 0))}GB",
                                'Storage': storage,
                                'Cpu_brand': example.get('Cpu brand', 'Unknown'),
                                'Gpu_brand': example.get('Gpu brand', 'Unknown'),
                                'Weight': f"{example.get('Weight', 0):.1f}kg" if example.get('Weight', 0) > 0 else "Weight N/A",
                                'Price': f"RS {example.get('Price', 0):,.2f}",
                                'Features': features_text,
                                'Touchscreen': 'Yes' if example.get('Touchscreen', 0) else 'No',
                                'Ips': 'Yes' if example.get('Ips', 0) else 'No',
                                'os': example.get('os', 'Unknown OS')
                            })
                    recommendations.extend(cluster_examples)
                recommendations = recommendations[:5]  # Limit to 5
            except Exception as e:
                print(f"Clustering error: {str(e)}")
                recommendations = []
                for _, row in filtered_df.nlargest(5, 'Ram').iterrows():
                    example = row.to_dict()
                    storage_parts = []
                    if example.get('SSD', 0) > 0:
                        storage_parts.append(f"{int(example['SSD'])}GB SSD")
                    if example.get('HDD', 0) > 0:
                        storage_parts.append(f"{int(example['HDD'])}GB HDD")
                    storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                    features = []
                    if example.get('Touchscreen', 0):
                        features.append('Touchscreen')
                    if example.get('Ips', 0):
                        features.append('IPS Display')
                    features_text = ', '.join(features) if features else 'Standard Features'
                    recommendations.append({
                        'Company': example.get('Company', 'Unknown'),
                        'TypeName': example.get('TypeName', 'Laptop'),
                        'Title': f"{example.get('Company', 'Unknown')} {example.get('TypeName', 'Laptop')}",
                        'Ram': f"{int(example.get('Ram', 0))}GB",
                        'Storage': storage,
                        'Cpu_brand': example.get('Cpu brand', 'Unknown'),
                        'Gpu_brand': example.get('Gpu brand', 'Unknown'),
                        'Weight': f"{example.get('Weight', 0):.1f}kg" if example.get('Weight', 0) > 0 else "Weight N/A",
                        'Price': f"RS {example.get('Price', 0):,.2f}",
                        'Features': features_text,
                        'Touchscreen': 'Yes' if example.get('Touchscreen', 0) else 'No',
                        'Ips': 'Yes' if example.get('Ips', 0) else 'No',
                        'os': example.get('os', 'Unknown OS')
                    })
        else:
            recommendations = []
            for _, row in filtered_df.nlargest(5, 'Ram').iterrows():
                example = row.to_dict()
                storage_parts = []
                if example.get('SSD', 0) > 0:
                    storage_parts.append(f"{int(example['SSD'])}GB SSD")
                if example.get('HDD', 0) > 0:
                    storage_parts.append(f"{int(example['HDD'])}GB HDD")
                storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                features = []
                if example.get('Touchscreen', 0):
                    features.append('Touchscreen')
                if example.get('Ips', 0):
                    features.append('IPS Display')
                features_text = ', '.join(features) if features else 'Standard Features'
                recommendations.append({
                    'Company': example.get('Company', 'Unknown'),
                    'TypeName': example.get('TypeName', 'Laptop'),
                    'Title': f"{example.get('Company', 'Unknown')} {example.get('TypeName', 'Laptop')}",
                    'Ram': f"{int(example.get('Ram', 0))}GB",
                    'Storage': storage,
                    'Cpu_brand': example.get('Cpu brand', 'Unknown'),
                    'Gpu_brand': example.get('Gpu brand', 'Unknown'),
                    'Weight': f"{example.get('Weight', 0):.1f}kg" if example.get('Weight', 0) > 0 else "Weight N/A",
                    'Price': f"RS {example.get('Price', 0):,.2f}",
                    'Features': features_text,
                    'Touchscreen': 'Yes' if example.get('Touchscreen', 0) else 'No',
                    'Ips': 'Yes' if example.get('Ips', 0) else 'No',
                    'os': example.get('os', 'Unknown OS')
                })

        return render_template('recommendations.html',
                              recommendations=recommendations,
                              budget=f"RS {budget:,.2f}")

    except Exception as e:
        flash(f"Error generating recommendations: {str(e)}", 'error')
        return redirect(url_for('index'))

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
    print("Accessing /signup route [GET or POST]")
    errors = {}
    form_data = {}
    success = False

    if request.method == 'POST':
        print("Received POST request for signup")
        try:
            form_data['username'] = request.form.get('username', '')
            form_data['email'] = request.form.get('email', '')
            form_data['password'] = request.form.get('password', '')
            form_data['confirmPassword'] = request.form.get('confirmPassword', '')
            print(f"Form data received: {request.form}")

            if not form_data['username'] or len(form_data['username']) < 8 or not re.match(r'^[A-Za-z][A-Za-z0-9]{7,19}$', form_data['username']):
                print("Username validation failed")
                errors['username'] = 'Username must be valid and at least 8 characters long'
            if not form_data['email'] or not re.match(r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$', form_data['email']):
                print("Email validation failed")
                errors['email'] = 'Invalid email address'
            if not form_data['password'] or len(form_data['password']) < 5 or not re.match(r'^[a-zA-Z0-9]{5,20}$', form_data['password']):
                print("Password validation failed")
                errors['password'] = 'Password must be at least 5 characters long'
            if not form_data['confirmPassword'] or form_data['password'] != form_data['confirmPassword']:
                print("Password mismatch")
                errors['confirmPassword'] = 'Passwords do not match'

            if errors:
                print("Validation errors detected, rendering signup.html")
                return render_template('signup.html', errors=errors, form_data=form_data, success=False)

            connection = create_connection()
            if not connection:
                print("Database connection failed")
                errors['general'] = 'Database connection error'
                return render_template('signup.html', errors=errors, form_data=form_data, success=False)

            try:
                cursor = connection.cursor()
                cursor.execute("SELECT uid FROM users WHERE username = %s OR email = %s", (form_data['username'], form_data['email']))
                if cursor.fetchone():
                    print("Duplicate username or email found")
                    errors['general'] = 'Username or email already exists'
                    return render_template('signup.html', errors=errors, form_data=form_data, success=False)

                hashed_password = generate_password_hash(form_data['password'], method='pbkdf2:sha256')
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (form_data['username'], form_data['email'], hashed_password)
                )
                connection.commit()
                print("User inserted successfully")
                flash('You have successfully signed up! Please log in.', 'success')
                return render_template('signup.html', errors={}, form_data={}, success=True)
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")
                errors['general'] = f'Database error: {str(db_error)}'
                return render_template('signup.html', errors=errors, form_data=form_data, success=False)
            finally:
                cursor.close()
                close_connection(connection)
        except Exception as e:
            print(f"Signup error: {str(e)}")
            errors['general'] = f'Error during signup: {str(e)}'
            return render_template('signup.html', errors=errors, form_data=form_data, success=False)

    print("Rendering signup.html for GET request")
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
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                users = cursor.fetchall()
                cursor.execute("SELECT * FROM predictions")
                predictions = cursor.fetchall()
        except Exception as e:
            print(f"Database error in admindashboard: {str(e)}")
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('admindashboard.html', users=users, predictions=predictions)

@app.route('/userlist')
def user_list():
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    users = []

    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                users = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('user_list.html', users=users)

@app.route('/userprediction')
def user_prediction():
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    predictions = []

    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
                predictions = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('user_prediction.html', predictions=predictions)

@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    errors = {}
    user = None

    if connection:
        try:
            with connection.cursor() as cursor:
                if request.method == 'POST':
                    username = request.form['username']
                    email = request.form['email']
                    password = request.form['password']
                    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

                    cursor.execute("SELECT uid FROM users WHERE (username = %s OR email = %s) AND uid != %s", (username, email, user_id))
                    conflict = cursor.fetchone()

                    if conflict:
                        errors['conflict'] = "Username or email already exists."
                        cursor.execute("SELECT * FROM users WHERE uid=%s", (user_id,))
                        user = cursor.fetchone()
                        return render_template('edit_user.html', user=user, errors=errors)

                    cursor.execute("UPDATE users SET username=%s, email=%s, password=%s WHERE uid=%s",
                                   (username, email, hashed_password, user_id))
                    connection.commit()
                    flash('User updated successfully', 'success')
                    return redirect(url_for('index'))

                cursor.execute("SELECT * FROM users WHERE uid=%s", (user_id,))
                user = cursor.fetchone()

                if not user:
                    flash('User not found', 'error')
                    return redirect(url_for('index'))
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('edit_user.html', user=user, errors=errors)

@app.route('/delete_user/<int:user_id>', methods=['GET'])
def delete_user(user_id):
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM predictions WHERE uid=%s", (user_id,))
                cursor.execute("DELETE FROM users WHERE uid=%s", (user_id,))
                connection.commit()
                flash('User and associated data deleted successfully', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('index'))

@app.route('/prediction_history')
def prediction_history():
    if 'user_id' not in session:
        print("No user_id in session, redirecting to login")
        flash('Please log in to view your prediction history.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    print(f"Accessing /prediction_history route for user_id: {user_id}")

    connection = create_connection()
    if not connection:
        print("Database connection failed")
        flash('Database connection error.', 'error')
        return redirect(url_for('dashboard'))

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            print(f"No user found for user_id: {user_id}")
            flash('User not found. Please log in again.', 'error')
            session.pop('user_id', None)
            return redirect(url_for('login'))

        username = user['username']
        print(f"Username fetched: {username}")

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
        print(f"Fetched {len(predictions)} predictions for user_id {user_id}")

        return render_template('predictionhistory.html', predictions=predictions, username=username)
    except Exception as e:
        print(f"Error fetching predictions for user_id {user_id}: {str(e)}")
        flash(f"Error fetching predictions: {str(e)}", 'error')
        return render_template('predictionhistory.html', predictions=[], username='User')
    finally:
        if 'cursor' in locals():
            cursor.close()
        close_connection(connection)

@app.route('/delete_prediction/<int:pid>', methods=['POST'])
def delete_prediction(pid):
    if 'user_id' not in session:
        print("No user_id in session, redirecting to login")
        flash('Please log in to delete predictions.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    print(f"Attempting to delete prediction with pid {pid} for user_id {user_id}")
    connection = create_connection()
    if not connection:
        print("Database connection failed")
        flash('Database connection error.', 'error')
        return redirect(url_for('prediction_history'))

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM predictions WHERE pid = %s AND uid = %s", (pid, user_id))
        if cursor.rowcount == 0:
            print(f"No prediction found with pid {pid} for user_id {user_id}")
            flash('Prediction not found or you do not have permission to delete it.', 'error')
        else:
            connection.commit()
            print(f"Prediction with pid {pid} deleted successfully")
            flash('Prediction deleted successfully.', 'success')
        return redirect(url_for('prediction_history'))
    except Exception as e:
        print(f"Error deleting prediction with pid {pid}: {str(e)}")
        flash(f"Error deleting prediction: {str(e)}", 'error')
        return redirect(url_for('prediction_history'))
    finally:
        if cursor:
            cursor.close()
        close_connection(connection)

@app.route('/view_prediction/<int:pid>')
def view_prediction(pid):
    if 'user_id' not in session:
        print("No user_id in session, redirecting to login")
        flash('Please log in to view predictions.', 'error')
        return redirect(url_for('login'))

    user_id = session['user_id']
    print(f"Accessing /view_prediction route for pid {pid} and user_id {user_id}")
    connection = create_connection()
    if not connection:
        print("Database connection failed")
        flash('Database connection error.', 'error')
        return redirect(url_for('prediction_history'))

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT pid, created_at, predicted_price, company, type, ram, weight, touchscreen, ips,
                   screen_size, resolution, cpu, hdd, ssd, gpu, os
            FROM predictions
            WHERE pid = %s AND uid = %s
        """, (pid, user_id))
        prediction = cursor.fetchone()
        if not prediction:
            print(f"No prediction found with pid {pid} for user_id {user_id}")
            flash('Prediction not found.', 'error')
            return redirect(url_for('prediction_history'))

        cursor.execute("SELECT username FROM users WHERE uid = %s", (user_id,))
        user = cursor.fetchone()
        username = user['username'] if user else 'User'
        print(f"Username fetched: {username}")

        return render_template('view_prediction.html', prediction=prediction, username=username)
    except Exception as e:
        print(f"Error fetching prediction with pid {pid}: {str(e)}")
        flash(f"Error fetching prediction: {str(e)}", 'error')
        return redirect(url_for('prediction_history'))
    finally:
        if cursor:
            cursor.close()
        close_connection(connection)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/save_recommendation', methods=['POST'])
def save_recommendation():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login to save recommendations'})

    try:
        data = request.get_json()
        laptop_name = data.get('laptop_name')
        specs = data.get('specs')
        price = data.get('price')
        similarity_score = data.get('similarity_score')

        connection = create_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO recommendations (uid, laptop_name, specs, price, similarity_score) VALUES (%s, %s, %s, %s, %s)",
                        (session['user_id'], laptop_name, specs, price, similarity_score)
                    )
                    connection.commit()
                    return jsonify({'success': True, 'message': 'Recommendation saved successfully'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error: {str(e)}'})
            finally:
                close_connection(connection)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/create_booking', methods=['POST'])
def create_booking():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login to create a booking'})

    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        product_name = data.get('product_name')
        specs = data.get('specs')
        price = data.get('price')

        connection = create_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO bookings (uid, prediction_id, product_name, specs, price) VALUES (%s, %s, %s, %s, %s)",
                        (session['user_id'], prediction_id, product_name, specs, price)
                    )
                    connection.commit()
                    return jsonify({'success': True, 'message': 'Booking created successfully'})
            except Exception as e:
                return jsonify({'success': False, 'message': f'Error: {str(e)}'})
            finally:
                close_connection(connection)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/recommendation_history')
def recommendation_history():
    if 'user_id' not in session:
        flash('Please login to view recommendation history', 'warning')
        return redirect(url_for('index'))

    connection = create_connection()
    recommendations = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM recommendations WHERE uid = %s ORDER BY saved_at DESC", (session['user_id'],))
                recommendations = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('recommendation_history.html', recommendations=recommendations)

@app.route('/booking_history')
def booking_history():
    if 'user_id' not in session:
        flash('Please login to view booking history', 'warning')
        return redirect(url_for('index'))

    connection = create_connection()
    bookings = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM bookings WHERE uid = %s ORDER BY created_at DESC", (session['user_id'],))
                bookings = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('booking_history.html', bookings=bookings)

@app.route('/all_recommendations')
def all_recommendations():
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    recommendations = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT r.*, u.username FROM recommendations r JOIN users u ON r.uid = u.uid ORDER BY r.saved_at DESC")
                recommendations = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('all_recommendations.html', recommendations=recommendations)

@app.route('/all_bookings')
def all_bookings():
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    bookings = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT b.*, u.username FROM bookings b JOIN users u ON b.uid = u.uid ORDER BY b.created_at DESC")
                bookings = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('all_bookings.html', bookings=bookings)

if __name__ == '__main__':
    app.run(debug=True)