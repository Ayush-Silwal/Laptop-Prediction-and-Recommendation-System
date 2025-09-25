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
    def __init__(self, max_depth=None, min_samples_split=5, min_samples_leaf=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features
        elif self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(n_features))
            
        self.feature_importances_ = np.zeros(n_features)
        self.tree_ = self._build_tree(X, y, depth=0)
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
    
    def _build_tree(self, X, y, depth):
        num_samples, n_features = X.shape
        
        # Enhanced stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           num_samples < self.min_samples_split or \
           num_samples < 2 * self.min_samples_leaf or \
           len(np.unique(y)) == 1 or \
           np.var(y) < 1e-7:
            return np.mean(y)

        # Feature sampling for better generalization
        if self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)

        best_split = self._find_best_split(X, y, feature_indices)
        if best_split is None:
            return np.mean(y)
        
        # Update feature importance
        self.feature_importances_[best_split['feature']] += best_split['importance']
        
        left_indices = X[:, best_split['feature']] <= best_split['value']
        right_indices = ~left_indices
        
        # Ensure minimum samples in each leaf
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            return np.mean(y)
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': best_split['feature'],
            'value': best_split['value'],
            'left': left_tree,
            'right': right_tree
        }
    
    def _find_best_split(self, X, y, feature_indices):
        best_split = None
        best_score = float('inf')
        current_mse = np.var(y)

        for feature in feature_indices:
            unique_values = np.unique(X[:, feature])
            if len(unique_values) == 1:
                continue
                
            # Smarter split point selection
            if len(unique_values) > 20:
                # Use quantiles for continuous features
                split_points = np.percentile(unique_values, [10, 25, 50, 75, 90])
            else:
                # Use midpoints for discrete features
                split_points = [(unique_values[i] + unique_values[i+1]) / 2 
                               for i in range(len(unique_values)-1)]
            
            for value in split_points:
                left_indices = X[:, feature] <= value
                right_indices = ~left_indices
                
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                # Weighted MSE
                left_weight = len(left_y) / len(y)
                right_weight = len(right_y) / len(y)
                weighted_mse = left_weight * np.var(left_y) + right_weight * np.var(right_y)
                
                if weighted_mse < best_score:
                    importance = current_mse - weighted_mse  # Information gain
                    best_split = {
                        'feature': feature, 
                        'value': value, 
                        'importance': importance
                    }
                    best_score = weighted_mse
        
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
    def __init__(self, n_estimators=150, max_depth=15, max_features='sqrt', 
                 min_samples_split=5, min_samples_leaf=2, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Better max_features calculation
        if self.max_features == 'sqrt':
            max_feats = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            max_feats = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            max_feats = max(1, int(self.max_features * n_features))
        else:
            max_feats = self.max_features or n_features
            
        feature_importance_sum = np.zeros(n_features)
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
                
            # Random feature selection
            feature_idx = np.random.choice(n_features, max_feats, replace=False)
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feats
            )
            
            X_sub = X_sample[:, feature_idx]
            tree.fit(X_sub, y_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
            
            # Accumulate feature importance
            if tree.feature_importances_ is not None:
                feature_importance_sum[feature_idx] += tree.feature_importances_
            
        # Normalize feature importance
        self.feature_importances_ = feature_importance_sum / self.n_estimators
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()
    
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
    def __init__(self, k=7, metric='hybrid', weights='distance'):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.feature_weights = None
        
    def set_feature_weights(self, weights):
        """Set importance weights for different features"""
        self.feature_weights = np.array(weights) if weights is not None else None
    
    def _cosine_similarity(self, a, b, weights=None):
        if weights is not None:
            a = a * weights
            b = b * weights
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _euclidean_distance(self, a, b, weights=None):
        if weights is not None:
            a = a * weights
            b = b * weights
        return np.sqrt(np.sum((a - b) ** 2))
    
    def _manhattan_distance(self, a, b, weights=None):
        if weights is not None:
            a = a * weights
            b = b * weights
        return np.sum(np.abs(a - b))
    
    def _hybrid_similarity(self, a, b, weights=None):
        """Combines multiple similarity metrics for better accuracy"""
        cos_sim = self._cosine_similarity(a, b, weights)
        euc_dist = self._euclidean_distance(a, b, weights)
        # Convert euclidean to similarity
        euc_sim = 1 / (1 + euc_dist)
        
        # Weighted combination - cosine is better for high-dimensional sparse data
        return 0.7 * cos_sim + 0.3 * euc_sim
    
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
            if self.metric == 'hybrid':
                similarities = [self._hybrid_similarity(sample, x, self.feature_weights) for x in self.X_train]
                neighbor_indices = np.argpartition(similarities, -self.k)[-self.k:]
                neighbor_similarities = [similarities[i] for i in neighbor_indices]
            elif self.metric == 'cosine':
                similarities = [self._cosine_similarity(sample, x, self.feature_weights) for x in self.X_train]
                neighbor_indices = np.argpartition(similarities, -self.k)[-self.k:]
                neighbor_similarities = [similarities[i] for i in neighbor_indices]
            else:  # euclidean or manhattan
                if self.metric == 'manhattan':
                    distances = [self._manhattan_distance(sample, x, self.feature_weights) for x in self.X_train]
                else:
                    distances = [self._euclidean_distance(sample, x, self.feature_weights) for x in self.X_train]
                neighbor_indices = np.argpartition(distances, self.k)[:self.k]
                neighbor_similarities = [1/(1+distances[i]) for i in neighbor_indices]  # Convert to similarities
            
            neighbor_values = self.y_train[neighbor_indices]
            
            # Apply distance weighting if requested
            if self.weights == 'distance':
                total_similarity = sum(neighbor_similarities)
                if total_similarity > 0:
                    weights_array = np.array(neighbor_similarities) / total_similarity
                    prediction = np.average(neighbor_values, weights=weights_array)
                else:
                    prediction = np.mean(neighbor_values)
            else:
                prediction = np.mean(neighbor_values)
                
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def get_similar_laptops(self, X_input, df, top_n=5, price_range_factor=0.3):
        """Enhanced similarity search with price filtering and diversity"""
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        if issparse(X_input):
            X_input = X_input.toarray()
            
        similarities = []
        for i, sample in enumerate(self.X_train):
            if self.metric == 'hybrid':
                sim = self._hybrid_similarity(X_input[0], sample, self.feature_weights)
            else:
                sim = self._cosine_similarity(X_input[0], sample, self.feature_weights)
            similarities.append((sim, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Get diverse recommendations
        recommendations = []
        seen_companies = set()
        seen_types = set()
        
        for sim_score, idx in similarities:
            if len(recommendations) >= top_n:
                break
                
            laptop = df.iloc[idx].copy()
            company = laptop.get('Company', 'Unknown')
            type_name = laptop.get('TypeName', 'Unknown')
            
            # Diversity constraints
            company_count = sum(1 for r in recommendations if r.get('Company') == company)
            type_key = f"{company}-{type_name}"
            
            if company_count >= 2 or type_key in seen_types:
                continue
                
            seen_companies.add(company)
            seen_types.add(type_key)
            
            # Build comprehensive laptop info
            ram = laptop.get('Ram', 0)
            ssd = laptop.get('SSD', 0)
            hdd = laptop.get('HDD', 0)
            cpu = laptop.get('Cpu brand', 'Unknown')
            gpu = laptop.get('Gpu brand', 'Unknown')
            weight = laptop.get('Weight', 0)
            price = laptop.get('Price', 0)
            
            # Storage description
            storage_parts = []
            if ssd > 0:
                storage_parts.append(f"{int(ssd)}GB SSD")
            if hdd > 0:
                storage_parts.append(f"{int(hdd)}GB HDD")
            storage = " + ".join(storage_parts) if storage_parts else "Storage info unavailable"
            
            # Feature classification
            features = []
            if laptop.get('Touchscreen', 0):
                features.append('Touchscreen')
            if laptop.get('Ips', 0):
                features.append('IPS Display')
            
            # Performance classification
            if ram >= 16 and ssd >= 512:
                features.append('High Performance')
            elif ram >= 8 and ssd >= 256:
                features.append('Mid Performance')
            else:
                features.append('Basic Performance')
            
            laptop_info = {
                'Company': company,
                'TypeName': type_name,
                'Title': f"{company} {type_name}",
                'Ram': f"{int(ram)}GB",
                'Storage': storage,
                'Cpu_brand': cpu,
                'Gpu_brand': gpu,
                'Weight': f"{weight:.1f}kg" if weight > 0 else "Weight N/A",
                'Price': float(price),
                'Similarity': f"{sim_score:.3f}",
                'Features': ', '.join(features) if features else 'Standard Features',
                'Touchscreen': 'Yes' if laptop.get('Touchscreen', 0) else 'No',
                'Ips': 'Yes' if laptop.get('Ips', 0) else 'No',
                'os': laptop.get('os', 'Unknown OS')
            }
            
            recommendations.append(laptop_info)
        
        return recommendations


class CustomKMeans:
    def __init__(self, n_clusters=5, max_iters=300, random_state=None, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
        # Enhanced cluster names based on laptop characteristics
        self.cluster_names = {
            0: "Budget-Friendly Everyday Laptops",
            1: "Mid-Range Professional Laptops", 
            2: "Premium Business Workstations",
            3: "Gaming & High-Performance Systems",
            4: "Ultraportable & Designer Laptops"
        }
        
    def _kmeans_plus_plus_init(self, X):
        """K-means++ initialization for better cluster centroids"""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.choice(n_samples)]
        
        for k in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = np.array([min([np.sum((x - c) ** 2) for c in centroids[:k]]) for x in X])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.random()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[k] = X[j]
                    break
                    
        return centroids
        
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
            
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Initialize centroids
        if self.init == 'k-means++':
            self.centroids = self._kmeans_plus_plus_init(X)
        else:
            idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[idx]
        
        prev_inertia = float('inf')
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            old_labels = self.labels_.copy() if self.labels_ is not None else np.zeros(n_samples)
            distances = self._calculate_distances(X)
            self.labels_ = np.argmin(distances, axis=1)
            
            # Calculate inertia
            self.inertia_ = np.sum([distances[i, self.labels_[i]] for i in range(n_samples)])
            
            # Check for convergence
            if np.all(old_labels == self.labels_) or abs(prev_inertia - self.inertia_) < 1e-6:
                break
                
            prev_inertia = self.inertia_
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    new_centroids[k] = np.mean(X[self.labels_ == k], axis=0)
                else:
                    # If cluster is empty, reinitialize
                    new_centroids[k] = X[np.random.choice(n_samples)]
            
            self.centroids = new_centroids
        
        return self
    
    def _calculate_distances(self, X):
        """Calculate distances from all points to all centroids"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        return distances
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if issparse(X):
            X = X.toarray()
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _analyze_cluster_characteristics(self, cluster_df):
        """Analyze cluster to determine appropriate name"""
        avg_price = cluster_df['Price'].mean()
        avg_ram = cluster_df['Ram'].mean()
        avg_weight = cluster_df['Weight'].mean()
        touchscreen_ratio = cluster_df['Touchscreen'].mean()
        gaming_ratio = cluster_df['TypeName'].str.contains('Gaming', case=False).mean()
        ultrabook_ratio = cluster_df['TypeName'].str.contains('Ultrabook', case=False).mean()
        workstation_ratio = cluster_df['TypeName'].str.contains('Workstation', case=False).mean()
        
        # Classification logic
        if gaming_ratio > 0.3:
            return "Gaming & High-Performance Systems"
        elif workstation_ratio > 0.2:
            return "Premium Business Workstations"
        elif ultrabook_ratio > 0.3 or (avg_weight < 2.0 and avg_price > 60000):
            return "Ultraportable & Designer Laptops"
        elif avg_price < 30000:
            return "Entry-Level Student Laptops"
        elif avg_price < 50000:
            return "Budget-Friendly Everyday Laptops"
        elif avg_price > 100000:
            return "Content Creation Powerhouses"
        elif avg_price > 80000:
            return "Premium Business Workstations"
        else:
            return "Mid-Range Professional Laptops"
    
    def get_cluster_examples(self, cluster_id, df, X_all=None, top_n=5):
        """Get diverse and representative examples from the cluster"""
        try:
            if X_all is not None:
                cluster_labels = self.predict(X_all)
            else:
                cluster_labels = self.labels_
                
            cluster_mask = cluster_labels == cluster_id
            cluster_df = df[cluster_mask].copy()
            
            if len(cluster_df) == 0:
                return []
            
            # Update cluster name based on actual data
            if cluster_id not in self.cluster_names or len(self.cluster_names) <= cluster_id:
                self.cluster_names[cluster_id] = self._analyze_cluster_characteristics(cluster_df)
            
            # Enhanced diversity scoring with more factors
            cluster_df['diversity_score'] = (
                cluster_df['Ram'] * 0.25 +                    # Memory importance
                cluster_df.get('SSD', 0) * 0.0001 +           # SSD storage
                cluster_df.get('ppi', 100) * 0.008 +          # Display quality
                (cluster_df['Weight'] * -3) +                 # Lighter is better
                cluster_df.get('Touchscreen', 0) * 8 +        # Premium features
                cluster_df.get('Ips', 0) * 8 +                # Display quality
                np.random.normal(0, 2, len(cluster_df))       # Add variety
            )
            
            # Sort by diversity and select varied examples
            cluster_df_sorted = cluster_df.sort_values(['diversity_score', 'Price'], 
                                                      ascending=[False, True])
            
            examples = []
            seen_companies = set()
            
            for _, laptop in cluster_df_sorted.iterrows():
                if len(examples) >= top_n:
                    break
                    
                company = laptop.get('Company', 'Unknown')
                
                # Ensure brand diversity
                if len(seen_companies) < 3 or company not in seen_companies:
                    seen_companies.add(company)
                    
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
                        storage_parts.append(f"{int(ssd)}GB SSD")
                    if hdd > 0:
                        storage_parts.append(f"{int(hdd)}GB HDD")
                    storage = " + ".join(storage_parts) if storage_parts else "No storage info"
                    
                    # Enhanced feature list
                    features = []
                    if laptop.get('Touchscreen', 0):
                        features.append('Touchscreen')
                    if laptop.get('Ips', 0):
                        features.append('IPS Display')
                    if ram >= 16:
                        features.append('High Memory')
                    if ssd >= 512:
                        features.append('Fast Storage')
                    if weight < 2.0:
                        features.append('Lightweight')
                        
                    features_text = ', '.join(features) if features else 'Standard Features'
                    
                    example = {
                        'Company': company,
                        'TypeName': type_name,
                        'Title': f"{company} {type_name}",
                        'Ram': f"{int(ram)}GB",
                        'Storage': storage,
                        'Cpu_brand': cpu,
                        'Gpu_brand': gpu,
                        'Weight': f"{weight:.1f}kg" if weight > 0 else "Weight N/A",
                        'Price': f"₹{price:,.2f}",  # Fixed currency symbol
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