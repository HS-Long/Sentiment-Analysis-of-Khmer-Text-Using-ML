"""
Machine learning models module for Khmer sentiment analysis
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import joblib


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y_train: Training labels
        
    Returns:
        dict: Dictionary mapping class indices to weights
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = {
        i: class_weights[i] for i in range(len(class_weights))
    }
    
    return class_weight_dict


def train_logistic_regression(X_train, y_train, class_weights=None, 
                               use_grid_search=True):
    """
    Train Logistic Regression model with optional grid search
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weights (dict, optional): Class weights
        use_grid_search (bool): Whether to use grid search
        
    Returns:
        Trained model
    """
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    log_reg = LogisticRegression(class_weight=class_weights, max_iter=1000)
    
    if use_grid_search:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "saga", "newton-cg"]
        }
        
        grid_search = GridSearchCV(
            log_reg,
            param_grid,
            scoring="f1_macro",
            cv=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search
    else:
        log_reg.fit(X_train, y_train)
        return log_reg


def train_svm(X_train, y_train, class_weights=None, use_grid_search=True):
    """
    Train SVM model with optional grid search
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weights (dict, optional): Class weights
        use_grid_search (bool): Whether to use grid search
        
    Returns:
        Trained model
    """
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    svm = SVC(class_weight=class_weights)
    
    if use_grid_search:
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
        
        grid_search = GridSearchCV(
            svm,
            param_grid,
            scoring="f1_macro",
            cv=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search
    else:
        svm.fit(X_train, y_train)
        return svm


def train_naive_bayes(X_train, y_train, use_grid_search=True):
    """
    Train Naive Bayes model with optional grid search
    
    Args:
        X_train: Training features
        y_train: Training labels
        use_grid_search (bool): Whether to use grid search
        
    Returns:
        Trained model
    """
    nb = MultinomialNB()
    
    if use_grid_search:
        param_grid = {
            "alpha": [0.1, 0.5, 1.0, 2.0]
        }
        
        grid_search = GridSearchCV(
            nb,
            param_grid,
            scoring="f1_macro",
            cv=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search
    else:
        nb.fit(X_train, y_train)
        return nb


def train_random_forest(X_train, y_train, class_weights=None, 
                       use_grid_search=True):
    """
    Train Random Forest model with optional grid search
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weights (dict, optional): Class weights
        use_grid_search (bool): Whether to use grid search
        
    Returns:
        Trained model
    """
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    rf = RandomForestClassifier(class_weight=class_weights, random_state=42)
    
    if use_grid_search:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        }
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            scoring="f1_macro",
            cv=3,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search
    else:
        rf.fit(X_train, y_train)
        return rf


def train_xgboost(X_train, y_train, class_weights=None, use_grid_search=True):
    """
    Train XGBoost model with optional grid search
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weights (dict, optional): Class weights
        use_grid_search (bool): Whether to use grid search
        
    Returns:
        Trained model
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
    
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    # Convert to sample weights for training
    train_weights = np.array([class_weights[y] for y in y_train])
    
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
    
    if use_grid_search:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3]
        }
        
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            scoring="f1_macro",
            cv=3,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train, sample_weight=train_weights)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search
    else:
        xgb.fit(X_train, y_train, sample_weight=train_weights)
        return xgb


def train_voting_classifier(X_train, y_train, models_dict):
    """
    Train a Voting Classifier ensemble
    
    Args:
        X_train: Training features
        y_train: Training labels
        models_dict (dict): Dictionary of (name, model) pairs
        
    Returns:
        Trained voting classifier
    """
    estimators = [(name, model.best_estimator_ if hasattr(model, 'best_estimator_') else model) 
                  for name, model in models_dict.items()]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='hard'
    )
    
    voting_clf.fit(X_train, y_train)
    return voting_clf


def save_model(model, filepath):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load a trained model from disk
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
