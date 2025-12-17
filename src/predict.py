"""
Utility script to make predictions with trained models
"""

import argparse
from data_preprocessing import khmer_preprocess_enhanced
from models import load_model


def predict_sentiment(text, vectorizer_path, model_path, label_encoder_path):
    """
    Predict sentiment for a given text
    
    Args:
        text (str): Input Khmer text
        vectorizer_path (str): Path to saved vectorizer
        model_path (str): Path to saved model
        label_encoder_path (str): Path to saved label encoder
        
    Returns:
        str: Predicted sentiment
    """
    # Load models
    vectorizer = load_model(vectorizer_path)
    model = load_model(model_path)
    label_encoder = load_model(label_encoder_path)
    
    # Preprocess text
    clean_text = khmer_preprocess_enhanced(text)
    
    # Transform to features
    X = vectorizer.transform([clean_text])
    
    # Predict
    prediction = model.predict(X)[0]
    sentiment = label_encoder.classes_[prediction]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = max(proba)
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}")
        print("\nProbabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name}: {proba[i]:.2%}")
    else:
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment}")
    
    return sentiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict sentiment for Khmer text')
    
    parser.add_argument('--text', type=str, required=True,
                       help='Khmer text to analyze')
    
    parser.add_argument('--vectorizer', type=str, 
                       default='models/vectorizer.pkl',
                       help='Path to saved vectorizer')
    
    parser.add_argument('--model', type=str, 
                       default='models/svm.pkl',
                       help='Path to saved model')
    
    parser.add_argument('--label-encoder', type=str, 
                       default='models/label_encoder.pkl',
                       help='Path to saved label encoder')
    
    args = parser.parse_args()
    
    predict_sentiment(args.text, args.vectorizer, args.model, args.label_encoder)
