# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import os

"""
Pokemon Legendary Classifier

This module implements a machine learning model to predict whether a Pokemon is legendary
based on its base stats and type. It uses a Random Forest classifier with SMOTE for
handling class imbalance.

Key Features:
- Feature engineering (total stats, physical/special averages)
- SMOTE for handling imbalanced classes
- Both single and batch prediction capabilities
- Comprehensive evaluation metrics

Example:
    >>> from pokemon_model import predict_legendary, rf_pipeline
    >>> pokemon = {
    ...     'hp': 100,
    ...     'attack': 150,
    ...     'defense': 140,
    ...     'sp_attack': 120,
    ...     'sp_defense': 100,
    ...     'speed': 90,
    ...     'type1': 'dragon'
    ... }
    >>> is_legendary, prob = predict_legendary(rf_pipeline, pokemon)
    >>> print(f"Legendary: {is_legendary}, Probability: {prob:.2%}")
"""

# Set random seed for reproducibility
np.random.seed(42)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(current_dir), 'data', 'pokemon.csv')

def load_and_prepare_data(data_path):
    """
    Load and prepare the Pokemon dataset for training.
    
    Args:
        data_path (str): Path to the Pokemon CSV file
        
    Returns:
        tuple: X_train, X_test, y_train, y_test splits of the data
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Feature engineering
    df['total_stats'] = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].sum(axis=1)
    df['physical_average'] = df[['attack', 'defense']].mean(axis=1)
    df['special_average'] = df[['sp_attack', 'sp_defense']].mean(axis=1)
    
    # Prepare features
    X_type = pd.get_dummies(df['type1'], prefix='type')
    X_numeric = df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 
                    'total_stats', 'physical_average', 'special_average']]
    X = pd.concat([X_numeric, X_type], axis=1)
    y = df['is_legendary']
    
    # Split data
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Evaluate model performance with various metrics and visualizations.
    
    Args:
        model: Trained sklearn model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name (str): Name of the model for display
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print(f"=== {model_name} Performance ===\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_pred_proba, model_name)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, X_train, model_name)

def predict_legendary(model, pokemon_stats):
    """
    Predict if a Pokemon is legendary based on its stats.
    
    Args:
        model: Trained model
        pokemon_stats (dict): Dictionary containing Pokemon stats
            Required keys: hp, attack, defense, sp_attack, sp_defense, speed, type1
            
    Returns:
        tuple: (is_legendary, probability)
            is_legendary (bool): True if predicted legendary
            probability (float): Probability of being legendary
    """
    # Create DataFrame
    pokemon_df = pd.DataFrame([pokemon_stats])
    
    # Feature engineering
    pokemon_df['total_stats'] = pokemon_df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].sum(axis=1)
    pokemon_df['physical_average'] = pokemon_df[['attack', 'defense']].mean(axis=1)
    pokemon_df['special_average'] = pokemon_df[['sp_attack', 'sp_defense']].mean(axis=1)
    
    # One-hot encode type
    type_dummies = pd.get_dummies(pokemon_df['type1'], prefix='type')
    
    # Add missing type columns
    for col in X_train.columns:
        if col.startswith('type_') and col not in type_dummies.columns:
            type_dummies[col] = 0
    
    # Combine features
    features = pd.concat([
        pokemon_df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
                    'total_stats', 'physical_average', 'special_average']],
        type_dummies
    ], axis=1)
    
    # Ensure columns match training data
    features = features[X_train.columns]
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return prediction, probability

def predict_multiple_pokemon(model, pokemon_list):
    """
    Predict legendary status for multiple Pokemon.
    
    Args:
        model: Trained model
        pokemon_list (list): List of dictionaries containing Pokemon stats
        
    Returns:
        list: List of dictionaries containing predictions and probabilities
    """
    results = []
    for pokemon in pokemon_list:
        prediction, probability = predict_legendary(model, pokemon)
        results.append({
            'stats': pokemon,
            'prediction': 'Legendary' if prediction else 'Not Legendary',
            'probability': probability
        })
    
    return results

# Load and prepare data
X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)

# Train Random Forest with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

rf_pipeline = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train_balanced, y_train_balanced)

# Optional: Evaluate model if running as main script
if __name__ == '__main__':
    evaluate_model(rf_pipeline, X_train_balanced, X_test, y_train_balanced, y_test, "Random Forest with SMOTE") 