from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from db_connection import create_connection, close_connection
import re
import pickle
import numpy as np
import pandas as pd
import joblib
from customModel import DecisionTree, RandomForest, CustomKNN, CustomKMeans
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the data
df = pickle.load(open("df.pkl", "rb"))

# Initialize models variable
pipe = None

# Load or train custom models
def load_or_train_models():
    global pipe
    model_path = "custom_models.pkl"
    
    if os.path.exists(model_path):
        try:
            print("Loading pre-trained custom models...")
            pipe = joblib.load(model_path)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            train_models()
    else:
        train_models()

def train_models():
    global pipe
    print("Training custom models...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Prepare data for training
        X = df.drop('Price', axis=1)
        y = np.log(df['Price'])
        
        # Identify categorical and numerical columns
        categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
        numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
        
        # Preprocessing pipeline
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Preprocess the data
        X_processed = preprocessor.fit_transform(X)
        
        # Train custom models
        print("Training Decision Tree...")
        dt_model = DecisionTree(max_depth=10)
        dt_model.fit(X_processed, y)
        
        print("Training Random Forest...")
        rf_model = RandomForest(n_estimators=50, max_depth=10)
        rf_model.fit(X_processed, y)
        
        print("Training KNN...")
        knn_model = CustomKNN(k=5, metric='cosine')
        knn_model.fit(X_processed, y)
        
        # Save models
        pipe = {
            'preprocessor': preprocessor,
            'decision_tree': dt_model,
            'random_forest': rf_model,
            'knn': knn_model
        }
        
        joblib.dump(pipe, "custom_models.pkl")
        print("Models trained and saved successfully!")
        
    except Exception as e:
        print(f"Error training models: {e}")
        # Fallback to simple average prediction
        pipe = {'fallback': True, 'avg_price': np.exp(np.mean(np.log(df['Price'])))}

# Load models on startup
load_or_train_models()

@app.route('/')
def index():
    return render_template('index.html', 
                           companies=df["Company"].unique(), 
                           types=df["TypeName"].unique(),
                           rams=df["Ram"].unique(),
                           cpus=df["Cpu brand"].unique(),
                           gpus=df["Gpu brand"].unique(),
                           oss=df["os"].unique())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        company = request.form.get('company')
        type_ = request.form.get('type')
        ram = int(request.form.get('ram'))
        weight = float(request.form.get('weight'))
        touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
        ips = 1 if request.form.get('ips') == 'Yes' else 0
        screen_size = float(request.form.get('screen_size'))
        resolution = request.form.get('resolution')
        cpu = request.form.get('cpu')
        hdd = int(request.form.get('HDD'))
        ssd = int(request.form.get('SSD'))
        gpu = request.form.get('gpu')
        os = request.form.get('os')

        # Calculate PPI
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Create a dataframe with the input data
        input_data = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen],
            'Ips': [ips],
            'ppi': [ppi],
            'Cpu brand': [cpu],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu brand': [gpu],
            'os': [os]
        })
        
        # Make prediction
        if 'fallback' in pipe and pipe['fallback']:
            predicted_price = int(pipe['avg_price'])
        else:
            # Preprocess the input data
            preprocessor = pipe['preprocessor']
            processed_input = preprocessor.transform(input_data)
            
            # Make prediction using custom Random Forest model
            predicted_log_price = pipe['random_forest'].predict(processed_input)[0]
            predicted_price = int(np.exp(predicted_log_price))

        # Return to index page with prediction result
        return render_template('index.html', 
                               companies=df["Company"].unique(), 
                               types=df["TypeName"].unique(),
                               rams=df["Ram"].unique(),
                               cpus=df["Cpu brand"].unique(),
                               gpus=df["Gpu brand"].unique(),
                               oss=df["os"].unique(),
                               predicted_price=predicted_price)
    
    except Exception as e:
        flash(f"Error making prediction: {str(e)}", 'error')
        return redirect(url_for('index'))

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
        
        # Apply filters
        filtered_df = filtered_df[filtered_df['Price'] <= budget]
        filtered_df = filtered_df[filtered_df['Ram'] >= min_ram]
        filtered_df = filtered_df[filtered_df['SSD'] >= min_ssd]
        
        if preferred_brands:
            filtered_df = filtered_df[filtered_df['Company'].isin(preferred_brands)]
        
        if preferred_cpu:
            filtered_df = filtered_df[filtered_df['Cpu brand'] == preferred_cpu]
        
        # If no results found, relax some filters
        if len(filtered_df) == 0:
            flash('No exact matches found. Showing closest options.', 'info')
            filtered_df = df.copy()
            filtered_df = filtered_df[filtered_df['Price'] <= budget * 1.2]
            filtered_df = filtered_df[filtered_df['Ram'] >= min_ram - 4]
        
        # Use KMeans clustering for recommendations
        if 'knn' in pipe and len(filtered_df) > 5:
            try:
                # Prepare features for clustering
                features = filtered_df[['Price', 'Ram', 'SSD', 'ppi']].copy()
                features = (features - features.mean()) / features.std()
                
                # Apply KMeans clustering
                kmeans = CustomKMeans(n_clusters=min(5, len(filtered_df)), random_state=42)
                clusters = kmeans.fit_predict(features)
                filtered_df['cluster'] = clusters
                
                # Get one laptop from each cluster
                recommendations = filtered_df.groupby('cluster').apply(lambda x: x.nlargest(1, 'Ram')).reset_index(drop=True)
            except:
                recommendations = filtered_df.nlargest(5, 'Ram')
        else:
            recommendations = filtered_df.nlargest(5, 'Ram')
        
        return render_template('recommendations.html', 
                              recommendations=recommendations.to_dict('records'),
                              budget=budget)
    
    except Exception as e:
        flash(f"Error generating recommendations: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        errors = {}

        # Server-side validation
        if not username:
            errors['username'] = "Please enter your username"
        if not password:
            errors['password'] = "Please enter your password"

        if errors:
            return render_template('login.html', errors=errors)

        # Authenticate user
        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                query = "SELECT * FROM users WHERE username = %s"
                cursor.execute(query, (username,))
                user = cursor.fetchone()
                cursor.close()

                if user and check_password_hash(user['password'], password):
                    # Successful login, set session
                    session['user_id'] = user['uid']
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
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        cpassword = request.form.get('confirmPassword')

        # Validation
        if len(username) < 8 or not re.match(r'^[A-Za-z][A-Za-z0-9]{7,19}$', username):
            flash('Username must be valid and at least 8 characters long', 'error')
            return redirect(url_for('signup'))

        if not re.match(r'^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$', email):
            flash('Invalid email address', 'error')
            return redirect(url_for('signup'))

        if len(password) < 5 or not re.match(r'^[a-zA-Z0-9]{5,20}$', password):
            flash('Password must be longer than five characters', 'error')
            return redirect(url_for('signup'))

        if password != cpassword:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                    (username, email, hashed_password)
                )   
                connection.commit()
                flash('You have successfully signed up!', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f"Error: {e}", 'danger')
                return redirect(url_for('signup'))
            finally:
                cursor.close()
                close_connection(connection)
        else:
            flash('Database connection error', 'error')

    return render_template('signup.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access dashboard', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Get form data
            company = request.form.get('company')
            type_ = request.form.get('type')
            ram = int(request.form.get('ram'))
            weight = float(request.form.get('weight'))
            touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
            ips = 1 if request.form.get('ips') == 'Yes' else 0
            screen_size = float(request.form.get('screen_size'))
            resolution = request.form.get('resolution')
            cpu = request.form.get('cpu')
            hdd = int(request.form.get('HDD'))
            ssd = int(request.form.get('SSD'))
            gpu = request.form.get('gpu')
            os = request.form.get('os')

            # Calculate PPI
            X_res, Y_res = map(int, resolution.split('x'))
            ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

            # Create a dataframe with the input data
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [type_],
                'Ram': [ram],
                'Weight': [weight],
                'Touchscreen': [touchscreen],
                'Ips': [ips],
                'ppi': [ppi],
                'Cpu brand': [cpu],
                'HDD': [hdd],
                'SSD': [ssd],
                'Gpu brand': [gpu],
                'os': [os]
            })
            
            # Make prediction
            if 'fallback' in pipe and pipe['fallback']:
                predicted_price = int(pipe['avg_price'])
            else:
                # Preprocess the input data
                preprocessor = pipe['preprocessor']
                processed_input = preprocessor.transform(input_data)
                
                # Make prediction using custom Random Forest model
                predicted_log_price = pipe['random_forest'].predict(processed_input)[0]
                predicted_price = int(np.exp(predicted_log_price))

            # Save prediction to database
            connection = create_connection()
            if connection:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO predictions (uid, company, type, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os, predicted_price) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                            (session['user_id'], company, type_, ram, weight, touchscreen, ips, screen_size, resolution, cpu, hdd, ssd, gpu, os, predicted_price))
                    connection.commit()
                    flash('Prediction saved successfully!', 'success')
                except Exception as e:
                    flash(f"An error occurred: {e}", 'error')
                finally:
                    close_connection(connection)

            return render_template('dashboard.html', 
                                   companies=df["Company"].unique(), 
                                   types=df["TypeName"].unique(),
                                   rams=df["Ram"].unique(),
                                   cpus=df["Cpu brand"].unique(),
                                   gpus=df["Gpu brand"].unique(),
                                   oss=df["os"].unique(),
                                   predicted_price=predicted_price)
        
        except Exception as e:
            flash(f"Error making prediction: {str(e)}", 'error')
            return render_template('dashboard.html', 
                                   companies=df["Company"].unique(), 
                                   types=df["TypeName"].unique(),
                                   rams=df["Ram"].unique(),
                                   cpus=df["Cpu brand"].unique(),
                                   gpus=df["Gpu brand"].unique(),
                                   oss=df["os"].unique())

    return render_template('dashboard.html', 
                           companies=df["Company"].unique(), 
                           types=df["TypeName"].unique(),
                           rams=df["Ram"].unique(),
                           cpus=df["Cpu brand"].unique(),
                           gpus=df["Gpu brand"].unique(),
                           oss=df["os"].unique())

@app.route('/predictionhistory')
def prediction_history():
    if 'user_id' not in session:
        flash('Please login to view prediction history', 'warning')
        return redirect(url_for('login'))

    connection = create_connection()
    history = []
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM predictions WHERE uid = %s ORDER BY created_at DESC", (session['user_id'],))
                history = cursor.fetchall()
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return render_template('predictionhistory.html', history=history)

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
                    
                    # Hash password
                    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

                    # Check if the username or email already exists in other records
                    cursor.execute("SELECT uid FROM users WHERE (username = %s OR email = %s) AND uid != %s", (username, email, user_id))
                    conflict = cursor.fetchone()

                    if conflict:
                        errors['conflict'] = "Username or email already exists."
                        # Get current user data to display in form
                        cursor.execute("SELECT * FROM users WHERE uid=%s", (user_id,))
                        user = cursor.fetchone()
                        return render_template('edit_user.html', user=user, errors=errors)

                    # Update user information if no conflict
                    cursor.execute("UPDATE users SET username=%s, email=%s, password=%s WHERE uid=%s",
                                   (username, email, hashed_password, user_id))
                    connection.commit()
                    flash('User updated successfully', 'success')
                    return redirect(url_for('user_list'))

                # GET request - fetch the current user data to display in the form
                cursor.execute("SELECT * FROM users WHERE uid=%s", (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    flash('User not found', 'error')
                    return redirect(url_for('user_list'))
                    
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
                # First delete user's predictions
                cursor.execute("DELETE FROM predictions WHERE uid=%s", (user_id,))
                # Then delete the user
                cursor.execute("DELETE FROM users WHERE uid=%s", (user_id,))
                connection.commit()
                flash('User and associated data deleted successfully', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('user_list'))

@app.route('/delete_prediction/<int:prediction_id>', methods=['GET'])
def delete_prediction(prediction_id):
    if 'admin_logged_in' not in session:
        flash('Access denied. Admin login required.', 'error')
        return redirect(url_for('admin_login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM predictions WHERE pid=%s", (prediction_id,))
                connection.commit()
                flash('Prediction deleted successfully', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('user_prediction'))

@app.route('/delete_prediction_user/<int:prediction_id>', methods=['GET'])
def delete_prediction_user(prediction_id):
    if 'user_id' not in session:
        flash('Please login to delete predictions', 'warning')
        return redirect(url_for('login'))

    connection = create_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM predictions WHERE pid=%s AND uid=%s", (prediction_id, session['user_id']))
                connection.commit()
                flash('Prediction deleted successfully', 'success')
        except Exception as e:
            flash(f"An error occurred: {e}", 'error')
        finally:
            close_connection(connection)

    return redirect(url_for('prediction_history'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)