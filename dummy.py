from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors

from dataTrain import DecisionTree, RandomForest
app = Flask(__name__)

# Initialize models (these would be trained when the server starts)
decision_tree = DecisionTree(max_depth=5)
random_forest = RandomForest(n_estimators=10, max_depth=5)
knn_recommender = KNNRecommender(k=5)

# Sample data for demonstration (replace with your actual data)
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train = np.random.rand(100)     # 100 target values
user_features = np.random.rand(50, 5)  # 50 users with 5 features each

@app.before_first_request
def train_models():
    """Train models when the server starts"""
    print("Training models...")
    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    knn_recommender.fit(user_features)
    print("Models trained successfully")

@app.route('/predict/decision_tree', methods=['POST'])
def predict_decision_tree():
    """Make predictions using the Decision Tree model"""
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = decision_tree.predict(features)
        return jsonify({
            'prediction': float(prediction[0]),
            'model': 'Decision Tree'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/random_forest', methods=['POST'])
def predict_random_forest():
    """Make predictions using the Random Forest model"""
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = random_forest.predict(features)
        return jsonify({
            'prediction': float(prediction[0]),
            'model': 'Random Forest'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get recommendations using KNN"""
    try:
        data = request.get_json()
        user_features = np.array(data['user_features']).reshape(1, -1)
        recommendations = knn_recommender.recommend(user_features)
        return jsonify({
            'recommendations': recommendations.tolist(),
            'model': 'KNN Recommender'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their parameters"""
    return jsonify({
        'models': {
            'DecisionTree': {'max_depth': decision_tree.max_depth},
            'RandomForest': {
                'n_estimators': random_forest.n_estimators,
                'max_depth': random_forest.max_depth
            },
            'KNNRecommender': {'k': knn_recommender.k}
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)