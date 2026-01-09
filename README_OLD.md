# Khmer Sentiment Analysis

A comprehensive machine learning system for sentiment analysis of Khmer (Cambodian) language text using multiple classification approaches, from traditional ML to deep learning models.

## 📋 Project Overview

This project implements an end-to-end sentiment analysis pipeline for Khmer text, classifying social media posts, reviews, and comments into three sentiment categories:
- **Positive** (វិជ្ជមាន)
- **Neutral** (អព្យាក្រឹត)
- **Negative** (អវិជ្ជមាន)

The system explores multiple machine learning approaches including:
- **Traditional ML**: Logistic Regression, SVM, Naive Bayes, Random Forest, XGBoost
- **Deep Learning**: Bidirectional LSTM with attention mechanisms
- **Ensemble Methods**: Voting classifiers for improved accuracy
- **REST API**: Flask-based API for real-time sentiment prediction

## 🎯 Key Features

### Advanced Text Processing
- **Khmer-Specific Preprocessing**: Unicode normalization (NFD→NFC), slang dictionary, special character handling
- **TF-IDF Vectorization**: N-gram features (unigrams, bigrams, trigrams) with customizable parameters
- **Class Imbalance Handling**: Automated class weight balancing for fair evaluation

### Multiple ML Approaches
- **Traditional ML Models**: Grid search hyperparameter optimization for each model
- **Deep Learning**: Bidirectional LSTM with dropout and early stopping
- **Threshold Optimization**: ROC curve analysis and optimal threshold selection

### Comprehensive Evaluation
- **Model Comparison**: Side-by-side performance metrics across all models
- **Confusion Matrices**: Visual representation of classification results
- **ROC Curves**: Multi-class ROC analysis with optimal threshold recommendations
- **Results Tracking**: Automated saving of model metadata and comparison reports

### Production-Ready API
- **Flask REST API**: Easy integration with web applications
- **Model Persistence**: Automatic loading of best performing models
- **Batch Processing**: Support for single text and batch predictions
- **Error Handling**: Robust error handling and validation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this project

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install XGBoost for advanced models:
```bash
pip install xgboost
```

## 📊 Dataset

The dataset should be in CSV format with the following columns:
- `text`: Khmer text to analyze
- `target`: Sentiment label (positive, neutral, negative)

Place your dataset in the `data/` directory.

## 🔧 Usage

### Training Models

Train all models with basic settings:
```bash
cd src
python train.py
```

Train specific models:
```bash
python train.py --models lr svm nb
```

Train with enhanced preprocessing:
```bash
python train.py --enhanced
```

Train with ensemble voting classifier:
```bash
python train.py --ensemble --save-models
```

Available options:
- `--data-path`: Path to dataset (default: `data/Data Collection - Sheet1.csv`)
- `--models`: Models to train: `all`, `lr`, `svm`, `nb`, `rf`, `xgb`
- `--enhanced`: Use enhanced preprocessing with stopwords removal
- `--ensemble`: Train voting ensemble classifier
- `--save-models`: Save trained models to disk
- `--model-dir`: Directory to save models (default: `models`)
- `--result-dir`: Directory to save results (default: `results`)

### Making Predictions

Predict sentiment for new text:
```bash
python predict.py --text "អរគុណច្រើន" --model models/svm.pkl
```

Options:
- `--text`: Khmer text to analyze (required)
- `--vectorizer`: Path to saved vectorizer (default: `models/vectorizer.pkl`)
- `--model`: Path to saved model (default: `models/svm.pkl`)
- `--label-encoder`: Path to saved label encoder (default: `models/label_encoder.pkl`)

## 📚 Module Documentation

### data_preprocessing.py

Functions for cleaning and preprocessing Khmer text:
- `khmer_preprocess(text)`: Basic text cleaning
- `khmer_preprocess_enhanced(text)`: Enhanced cleaning with stopword removal
- `load_and_clean_data(filepath)`: Load and preprocess dataset
- `encode_labels(df)`: Encode sentiment labels to numeric values

### feature_extraction.py

TF-IDF feature extraction:
- `create_tfidf_vectorizer()`: Create standard vectorizer
- `create_enhanced_vectorizer()`: Create enhanced vectorizer with optimized parameters
- `extract_features(df)`: Extract TF-IDF features from text
- `split_data(X, y)`: Split data into train/test sets

### models.py

Model training functions:
- `train_logistic_regression()`: Train Logistic Regression with grid search
- `train_svm()`: Train Support Vector Machine
- `train_naive_bayes()`: Train Multinomial Naive Bayes
- `train_random_forest()`: Train Random Forest classifier
- `train_xgboost()`: Train XGBoost classifier
- `train_voting_classifier()`: Train ensemble voting classifier
- `save_model()` / `load_model()`: Save/load trained models

### evaluation.py

Model evaluation and visualization:
- `evaluate_model()`: Calculate accuracy, F1 score, and print classification report
- `plot_confusion_matrix()`: Visualize confusion matrix
- `plot_sentiment_distribution()`: Plot label distribution
- `compare_models()`: Compare multiple models and visualize results
- `save_results()`: Save evaluation results to CSV

## 📈 Model Performance

Based on the experimental results:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~46% |
| SVM | ~52% |
| Naive Bayes | ~49% |
| Random Forest | ~50-55% |
| XGBoost | ~50-55% |
| Voting Classifier | ~52-56% |

*Note: Results may vary based on dataset and hyperparameters*

## 🔍 Key Features

### Text Preprocessing
- Unicode normalization for Khmer characters
- Removal of numbers and punctuation
- Khmer stopwords removal (enhanced mode)
- Whitespace normalization

### Feature Engineering
- TF-IDF vectorization with n-grams (unigrams, bigrams, trigrams)
- Configurable feature limits and document frequency filters
- Sublinear TF scaling for better performance

### Model Training
- Automated class weight balancing for imbalanced datasets
- Grid search hyperparameter optimization
- Cross-validation for robust evaluation
- Support for multiple model architectures

### Evaluation
- Comprehensive metrics (accuracy, F1-score, precision, recall)
- Confusion matrix visualization
- Model comparison charts
- Results export to CSV

## 💡 Improvement Strategies

To improve model performance:

1. **Collect More Data**: Sentiment analysis benefits greatly from larger, diverse datasets
2. **Advanced Preprocessing**: Implement Khmer-specific stemming/lemmatization
3. **Deep Learning**: Use pre-trained models like XLM-RoBERTa for multilingual understanding
4. **Feature Engineering**: Add character-level features, text length, sentiment lexicons
5. **Data Augmentation**: Use back-translation, synonym replacement



---

**Note**: This project is designed for Khmer language sentiment analysis. Results may vary based on dataset quality, size, and domain specificity.
