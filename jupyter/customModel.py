import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import resample
from scipy.sparse import issparse

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        self.tree_ = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           num_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
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
        
        return {
            'feature': best_split['feature'],
            'value': best_split['value'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _find_best_split(self, X, y):
        best_split = None
        best_mse = float('inf')
        num_features = X.shape[1]

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            split_points = np.percentile(unique_values, [25, 50, 75]) if len(unique_values) > 10 else unique_values
            
            for value in split_points:
                left_indices = X[:, feature] <= value
                right_indices = ~left_indices
                
                if np.sum(left_indices) < 2 or np.sum(right_indices) < 2:
                    continue
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                mse = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / len(y)
                
                if mse < best_mse:
                    best_split = {'feature': feature, 'value': value}
                    best_mse = mse
        
        return best_split

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
            
        return np.array([self._predict(sample, self.tree_) for sample in X])
    
    def _predict(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        if sample[tree['feature']] <= tree['value']:
            return self._predict(sample, tree['left'])
        return self._predict(sample, tree['right'])

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        n_features = X.shape[1]
        max_feats = int(np.sqrt(n_features)) if self.max_features == 'sqrt' else self.max_features
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)
            feature_idx = np.random.choice(n_features, max_feats, replace=False)
            X_sub = X_sample[:, feature_idx]
            
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sub, y_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
            
        all_preds = np.zeros((self.n_estimators, X.shape[0]))
        
        for i, (tree, feat_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_sub = X[:, feat_idx]
            all_preds[i] = tree.predict(X_sub)
            
        return np.mean(all_preds, axis=0)

class CustomKNN:
    def __init__(self, k=5, metric='cosine'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        
    def _cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0  # Return 0 similarity if either vector has zero norm
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
        self.X_train = X
        self.y_train = y.values if isinstance(y, pd.Series) else y
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
            
        predictions = []
        for sample in X:
            distances = []
            if self.metric == 'cosine':
                distances = [self._cosine_similarity(sample, x) for x in self.X_train]
                neighbors = np.argpartition(distances, -self.k)[-self.k:]
            else:
                distances = [self._euclidean_distance(sample, x) for x in self.X_train]
                neighbors = np.argpartition(distances, self.k)[:self.k]
            
            prediction = np.mean(self.y_train[neighbors])
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def get_similar_laptops(self, X_input, df, top_n=5):
        """Get similar laptops with better descriptions"""
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        if issparse(X_input):
            X_input = X_input.toarray()
            
        similarities = []
        for i, sample in enumerate(self.X_train):
            sim = self._cosine_similarity(X_input[0], sample)
            similarities.append((sim, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:top_n]]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            laptop = df.iloc[idx].copy()
            
            # Create better laptop description
            company = laptop.get('Company', 'Unknown')
            type_name = laptop.get('TypeName', 'Laptop')
            ram = laptop.get('Ram', 0)
            ssd = laptop.get('SSD', 0)
            hdd = laptop.get('HDD', 0)
            cpu = laptop.get('Cpu brand', 'Unknown')
            gpu = laptop.get('Gpu brand', 'Unknown')
            weight = laptop.get('Weight', 0)
            price = laptop.get('Price', 0)
            
            # Build storage description
            storage_parts = []
            if ssd > 0:
                storage_parts.append(f"{ssd}GB SSD")
            if hdd > 0:
                storage_parts.append(f"{hdd}GB HDD")
            storage = " + ".join(storage_parts) if storage_parts else "Storage info unavailable"
            
            # Enhanced laptop info
            laptop_info = {
                'Company': company,
                'TypeName': type_name,
                'Title': f"{company} {type_name}",
                'Ram': f"{ram}GB",
                'Storage': storage,
                'Cpu_brand': cpu,
                'Gpu_brand': gpu,
                'Weight': f"{weight:.1f}kg" if weight > 0 else "Weight N/A",
                'Price': price,
                'Similarity': f"{similarities[i][0]:.2f}",
                'Features': []
            }
            
            # Add features
            if laptop.get('Touchscreen', 0):
                laptop_info['Features'].append('Touchscreen')
            if laptop.get('Ips', 0):
                laptop_info['Features'].append('IPS Display')
            
            laptop_info['Features'] = ', '.join(laptop_info['Features']) if laptop_info['Features'] else 'Standard Features'
            
            recommendations.append(laptop_info)
        
        return recommendations

class CustomKMeans:
    def __init__(self, n_clusters=5, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        
        # Define meaningful cluster names
        self.cluster_names = {
            0: "Budget-Friendly Laptops",
            1: "Mid-Range Performance",
            2: "Premium Workstations",
            3: "Gaming & High-Performance", 
            4: "Ultraportable & Business"
        }
        
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
            
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Initialize centroids randomly
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            old_labels = self.labels_ if self.labels_ is not None else np.zeros(n_samples)
            self.labels_ = self._assign_clusters(X)
            
            # Check for convergence
            if np.all(old_labels == self.labels_):
                break
                
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels_ == k], axis=0)
        
        return self
    
    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sqrt(np.sum((X - self.centroids[k]) ** 2, axis=1))
        return np.argmin(distances, axis=1)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
        return self._assign_clusters(X)
    
    def get_cluster_examples(self, cluster_id, df, X_all, top_n=5):
        """Get diverse and representative examples from the cluster"""
        try:
            cluster_labels = self.predict(X_all)
            cluster_mask = cluster_labels == cluster_id
            cluster_df = df[cluster_mask].copy()
            
            if len(cluster_df) == 0:
                return []
            
            # Calculate diversity score for better representation
            cluster_df['diversity_score'] = (
                cluster_df['Ram'] * 0.3 +
                cluster_df['SSD'] * 0.0002 +  # Scale down storage
                cluster_df['ppi'] * 0.01 +    # Scale down PPI
                (cluster_df['Weight'] * -2) +  # Lower weight is better
                cluster_df['Touchscreen'] * 10 +
                cluster_df['Ips'] * 10
            )
            
            # Sort by diversity score and price for variety
            cluster_df_sorted = cluster_df.sort_values(['diversity_score', 'Price'], 
                                                      ascending=[False, True])
            
            examples = []
            for _, laptop in cluster_df_sorted.head(top_n).iterrows():
                company = laptop.get('Company', 'Unknown')
                type_name = laptop.get('TypeName', 'Laptop')
                ram = laptop.get('Ram', 0)
                ssd = laptop.get('SSD', 0)
                hdd = laptop.get('HDD', 0)
                cpu = laptop.get('Cpu brand', 'Unknown')
                gpu = laptop.get('Gpu brand', 'Unknown')
                weight = laptop.get('Weight', 0)
                price = laptop.get('Price', 0)
                
                # Build storage info
                storage_parts = []
                if ssd > 0:
                    storage_parts.append(f"{ssd}GB SSD")
                if hdd > 0:
                    storage_parts.append(f"{hdd}GB HDD")
                storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                
                # Build feature list
                features = []
                if laptop.get('Touchscreen', 0):
                    features.append('Touchscreen')
                if laptop.get('Ips', 0):
                    features.append('IPS Display')
                features_text = ', '.join(features) if features else 'Standard Features'
                
                example = {
                    'Company': company,
                    'TypeName': type_name,
                    'Title': f"{company} {type_name}",
                    'Ram': f"{ram}GB",
                    'Storage': storage,
                    'Cpu_brand': cpu,
                    'Gpu_brand': gpu,
                    'Weight': f"{weight:.1f}kg" if weight > 0 else "Weight N/A",
                    'Price': f"RS {price:,.2f}",
                    'Features': features_text,
                    'Touchscreen': 'Yes' if laptop.get('Touchscreen', 0) else 'No',
                    'Ips': 'Yes' if laptop.get('Ips', 0) else 'No',
                    'os': laptop.get('os', 'Unknown OS')
                }
                
                examples.append(example)
            
            return examples
            
        except Exception as e:
            print(f"Error getting cluster examples: {e}")
            return []