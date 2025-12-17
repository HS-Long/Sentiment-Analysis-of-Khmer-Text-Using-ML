"""
Main training script for Khmer sentiment analysis
"""

import os
import argparse
from data_preprocessing import load_and_clean_data, encode_labels
from feature_extraction import extract_features, split_data
from models import (
    train_logistic_regression, 
    train_svm, 
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
    train_voting_classifier,
    save_model
)
from evaluation import (
    evaluate_model, 
    plot_confusion_matrix, 
    compare_models,
    save_results
)


def main(args):
    """
    Main training pipeline
    """
    print("="*60)
    print("Khmer Sentiment Analysis - Training Pipeline")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = load_and_clean_data(args.data_path, use_enhanced=args.enhanced)
    df, label_encoder = encode_labels(df)
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {label_encoder.classes_}")
    
    # 2. Extract features
    print("\n[2/6] Extracting features...")
    X, vectorizer = extract_features(df, enhanced=args.enhanced)
    y = df["label"]
    print(f"Feature matrix shape: {X.shape}")
    
    # 3. Split data
    print("\n[3/6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4. Train models
    print("\n[4/6] Training models...")
    results = {}
    trained_models = {}
    
    if 'lr' in args.models or 'all' in args.models:
        print("\nTraining Logistic Regression...")
        lr_model = train_logistic_regression(X_train, y_train)
        lr_results = evaluate_model(lr_model, X_test, y_test, label_encoder, "Logistic Regression")
        results['Logistic Regression'] = lr_results['accuracy']
        trained_models['lr'] = lr_model
        
        if args.save_models:
            save_model(lr_model, os.path.join(args.model_dir, 'logistic_regression.pkl'))
    
    if 'svm' in args.models or 'all' in args.models:
        print("\nTraining SVM...")
        svm_model = train_svm(X_train, y_train)
        svm_results = evaluate_model(svm_model, X_test, y_test, label_encoder, "SVM")
        results['SVM'] = svm_results['accuracy']
        trained_models['svm'] = svm_model
        
        if args.save_models:
            save_model(svm_model, os.path.join(args.model_dir, 'svm.pkl'))
    
    if 'nb' in args.models or 'all' in args.models:
        print("\nTraining Naive Bayes...")
        nb_model = train_naive_bayes(X_train, y_train)
        nb_results = evaluate_model(nb_model, X_test, y_test, label_encoder, "Naive Bayes")
        results['Naive Bayes'] = nb_results['accuracy']
        trained_models['nb'] = nb_model
        
        if args.save_models:
            save_model(nb_model, os.path.join(args.model_dir, 'naive_bayes.pkl'))
    
    if 'rf' in args.models or 'all' in args.models:
        print("\nTraining Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        rf_results = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
        results['Random Forest'] = rf_results['accuracy']
        trained_models['rf'] = rf_model
        
        if args.save_models:
            save_model(rf_model, os.path.join(args.model_dir, 'random_forest.pkl'))
    
    if 'xgb' in args.models or 'all' in args.models:
        print("\nTraining XGBoost...")
        try:
            xgb_model = train_xgboost(X_train, y_train)
            xgb_results = evaluate_model(xgb_model, X_test, y_test, label_encoder, "XGBoost")
            results['XGBoost'] = xgb_results['accuracy']
            trained_models['xgb'] = xgb_model
            
            if args.save_models:
                save_model(xgb_model, os.path.join(args.model_dir, 'xgboost.pkl'))
        except ImportError as e:
            print(f"Skipping XGBoost: {e}")
    
    # 5. Train ensemble if requested
    if args.ensemble and len(trained_models) >= 2:
        print("\n[5/6] Training Voting Classifier...")
        voting_model = train_voting_classifier(X_train, y_train, trained_models)
        voting_results = evaluate_model(voting_model, X_test, y_test, label_encoder, "Voting Classifier")
        results['Voting Classifier'] = voting_results['accuracy']
        
        if args.save_models:
            save_model(voting_model, os.path.join(args.model_dir, 'voting_classifier.pkl'))
    
    # 6. Compare models and save results
    print("\n[6/6] Comparing models...")
    comparison = compare_models(results, save_path=os.path.join(args.result_dir, 'model_comparison.png'))
    save_results(results, os.path.join(args.result_dir, 'results.csv'))
    
    # Save vectorizer and label encoder
    if args.save_models:
        save_model(vectorizer, os.path.join(args.model_dir, 'vectorizer.pkl'))
        save_model(label_encoder, os.path.join(args.model_dir, 'label_encoder.pkl'))
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis models')
    
    parser.add_argument('--data-path', type=str, 
                       default='../data/Data Collection - Sheet1.csv',
                       help='Path to the dataset CSV file')
    
    parser.add_argument('--models', nargs='+', 
                       default=['all'],
                       choices=['all', 'lr', 'svm', 'nb', 'rf', 'xgb'],
                       help='Models to train (all, lr, svm, nb, rf, xgb)')
    
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced preprocessing and features')
    
    parser.add_argument('--ensemble', action='store_true',
                       help='Train voting ensemble classifier')
    
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models to disk')
    
    parser.add_argument('--model-dir', type=str, 
                       default='../models',
                       help='Directory to save models')
    
    parser.add_argument('--result-dir', type=str, 
                       default='../results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    main(args)
