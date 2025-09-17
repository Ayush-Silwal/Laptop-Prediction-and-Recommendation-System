#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. LIBRARY IMPORTS AND SETUP
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity

print("Libraries imported successfully.")


# In[2]:


# 2. DATA LOADING AND PREPROCESSING
try:
    df = pd.read_csv("car_dataset.csv")
    print("Dataset 'car_dataset.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'car_dataset.csv' not found. Please ensure the dataset is in the same directory.")
    exit()

# Create a clean copy of the original dataframe
original_df = df.copy()

# Define features and target
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Define feature types
cat_features = [col for col in X.columns if X[col].dtype == 'object']
num_features = X.select_dtypes(exclude="object").columns
onehot_columns = ['seller_type', 'fuel_type', 'transmission_type']
binary_columns = ['car_name']

# Preprocessing pipelines
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(handle_unknown='ignore')
binary_transformer = BinaryEncoder()

# ColumnTransformer setup
preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features),
        ("BinaryEncoder", binary_transformer, binary_columns)
    ],
    remainder='drop'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print(f"Data preprocessed. Training shape: {X_train_transformed.shape}, Testing shape: {X_test_transformed.shape}")


# In[3]:


# 3. HELPER FUNCTIONS
def evaluate_model(true, predicted):
    """Calculates and returns MAE, RMSE, and R2 Score."""
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[4]:


# 4. DECISION TREE IMPLEMENTATION
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        self.tree_ = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, _ = X.shape
        if num_samples <= 2 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)
        
        left_indices = X[:, best_split['feature']] <= best_split['value']
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.mean(y)

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}
    
    def _find_best_split(self, X, y):
        best_split, best_mse = None, float('inf')
        num_samples, num_features = X.shape

        for feature in range(num_features):
            values = np.unique(X[:, feature])
            for value in values:
                left_indices = X[:, feature] <= value
                right_indices = ~left_indices
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0: continue
                
                left_y, right_y = y[left_indices], y[right_indices]
                mse = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / num_samples
                
                if mse < best_mse:
                    best_split, best_mse = {'feature': feature, 'value': value}, mse
        
        return best_split

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        return np.array([self._predict(sample, self.tree_) for sample in X])
    
    def _predict(self, sample, tree):
        if not isinstance(tree, dict): return tree
        
        if sample[tree['feature']] <= tree['value']:
            return self._predict(sample, tree['left'])
        else:
            return self._predict(sample, tree['right'])

print("DecisionTree class defined.")


# In[5]:


# 5. RANDOM FOREST IMPLEMENTATION (DEPENDS ON SECTION 4)
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        self.trees = []
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_resampled, y_resampled)
            self.trees.append(tree)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

print("RandomForest class defined.")


# In[6]:


# 6. KNN IMPLEMENTATION (DEPENDS ON SECTION 1)
class KNeighborsRegressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        if hasattr(X, "toarray"): 
            X = X.toarray()
        elif isinstance(X, pd.DataFrame): 
            X = X.values
        if isinstance(y, pd.Series): 
            y = y.values
        self.X_train, self.y_train = X, y

    def _predict_single(self, x):
        x = np.array(x)
        distances = [np.sqrt(np.sum((x_train - x)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.mean(k_nearest_labels)

    def predict(self, X):
        if hasattr(X, "toarray"): 
            X = X.toarray()
        elif isinstance(X, pd.DataFrame): 
            X = X.values
        return np.array([self._predict_single(x) for x in X])

print("KNeighborsRegressor class defined.")


# In[7]:


# 7. MODEL TRAINING AND EVALUATION
models = {
    "Decision Tree": DecisionTree(max_depth=5),
    "Random Forest Regressor": RandomForest(n_estimators=100, max_depth=10),
    "K-Nearest Neighbors": KNeighborsRegressor(k=7)
}

results = []

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train_transformed, y_train)
    y_train_pred = model.predict(X_train_transformed)
    y_test_pred = model.predict(X_test_transformed)

    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(f"Results for {model_name}:")
    print(f"  Training Set -> MAE: {model_train_mae:.2f}, RMSE: {model_train_rmse:.2f}, R2: {model_train_r2:.4f}")
    print(f"  Test Set     -> MAE: {model_test_mae:.2f}, RMSE: {model_test_rmse:.2f}, R2: {model_test_r2:.4f}")
    
    results.append({
        'model': model_name, 'test_mae': model_test_mae, 'test_rmse': model_test_rmse, 'test_r2': model_test_r2
    })

print("\n===== Final Model Comparison (Test Set) =====")
results_df = pd.DataFrame(results)
print(results_df)

# Save the best model
best_model = models['Random Forest Regressor']
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(best_model, 'car_price_model.pkl')
print("\nModels saved to 'preprocessor.pkl' and 'car_price_model.pkl'.")


# In[8]:


# 8. SIMILARITY AND RECOMMENDATION SYSTEM (DEPENDS ON SECTIONS 1-2)
def custom_cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / ((norm_a * norm_b) + 1e-8)

class EnsembleSimilarity:
    def __init__(self, n_estimators=100, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.X, self.n_total_features, self.n_sub_features = None, None, None

    def fit(self, X):
        self.X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self.n_total_features = self.X.shape[1]
        if isinstance(self.max_features, int): self.n_sub_features = self.max_features
        elif isinstance(self.max_features, float): self.n_sub_features = int(self.n_total_features * self.max_features)
        else: self.n_sub_features = int(np.sqrt(self.n_total_features))
        self.n_sub_features = max(1, self.n_sub_features)

    def get_similarity_score(self, idx_a, idx_b):
        vec_a, vec_b = self.X[idx_a], self.X[idx_b]
        all_scores = []
        for _ in range(self.n_estimators):
            feature_indices = np.random.choice(self.n_total_features, self.n_sub_features, replace=False)
            sub_vec_a, sub_vec_b = vec_a[feature_indices], vec_b[feature_indices]
            all_scores.append(custom_cosine_similarity(sub_vec_a, sub_vec_b))
        return np.mean(all_scores)

    def get_similar_items(self, item_index, top_n=5):
        print(f"\nCalculating ensemble similarity for item {item_index}...")
        sim_scores = [(i, self.get_similarity_score(item_index, i)) for i in range(self.X.shape[0]) if i != item_index]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return [i[0] for i in sim_scores[:top_n]]

print("\nSimilarity functions defined.")


# In[9]:


# 9. RECOMMENDATION DEMO (DEPENDS ON ALL PREVIOUS SECTIONS)
# Process full dataset for recommendations
X_processed_full = preprocessor.transform(original_df.drop('selling_price', axis=1))
ensemble_sim = EnsembleSimilarity(n_estimators=50, max_features=0.5)
ensemble_sim.fit(X_processed_full)

def recommend_by_ensemble_features(car_name, top_n=5):
    try:
        car_index = original_df[original_df['car_name'] == car_name].index[0]
    except IndexError:
        return f"Car '{car_name}' not found."
    
    similar_indices = ensemble_sim.get_similar_items(car_index, top_n=top_n)
    print(f"\n--- Top {top_n} Cars Similar to '{car_name}' ---")
    return original_df.iloc[similar_indices]

def recommend_by_price(predicted_price, top_n=5, tolerance=0.15):
    lower_bound = predicted_price * (1 - tolerance)
    upper_bound = predicted_price * (1 + tolerance)
    
    recommended_cars = original_df[
        (original_df['selling_price'] >= lower_bound) & 
        (original_df['selling_price'] <= upper_bound)
    ].copy()
    
    recommended_cars['price_diff'] = abs(recommended_cars['selling_price'] - predicted_price)
    recommended_cars = recommended_cars.sort_values('price_diff')
    
    print(f"\n--- Top {top_n} Cars Around ₹{predicted_price:,.2f} (±{tolerance*100}%) ---")
    return recommended_cars.head(top_n).drop('price_diff', axis=1)

# Demo examples
print("\n=== DEMO 1: Feature-Based Recommendation ===")
print(recommend_by_ensemble_features("Hyundai i20 Sportz"))

print("\n=== DEMO 2: Price-Based Recommendation ===")
sample_car = X_test.iloc[[20]]
sample_transformed = preprocessor.transform(sample_car)
predicted_price = best_model.predict(sample_transformed)[0]
print(recommend_by_price(predicted_price))

print("\nScript execution completed.")


# In[ ]:




