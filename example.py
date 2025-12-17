# Example usage script

import sys
sys.path.append('src')

from data_preprocessing import load_and_clean_data, encode_labels
from feature_extraction import extract_features, split_data
from models import train_svm, calculate_class_weights
from evaluation import evaluate_model, plot_confusion_matrix

# Load and preprocess data
print("Loading data...")
df = load_and_clean_data('data/Data Collection - Sheet1.csv', use_enhanced=True)
df, label_encoder = encode_labels(df)

# Extract features
print("Extracting features...")
X, vectorizer = extract_features(df, enhanced=True)
y = df["label"]

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = split_data(X, y)

# Train model
print("Training SVM model...")
model = train_svm(X_train, y_train, use_grid_search=True)

# Evaluate
print("Evaluating model...")
results = evaluate_model(model, X_test, y_test, label_encoder, "SVM")

# Plot confusion matrix
plot_confusion_matrix(y_test, results['predictions'], 
                     results['target_names'], "SVM")

print("\nDone!")
