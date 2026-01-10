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

## 📁 Project Structure

```
PROJECT/
├── app.py                      # Flask REST API server
├── train.py                    # Main training pipeline
├── predict.py                  # Command-line prediction script
├── test_api.py                 # API testing utilities
├── setup.py                    # Package setup configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   ├── Data Collection - Sheet1.csv  # Original dataset
│   ├── data_cleaned_all.csv          # Preprocessed data (~1,057 samples)
│   └── new.csv                       # Additional data
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration settings
│   ├── data_loader.py                # Data loading and cleaning
│   ├── preprocessing.py              # Khmer text preprocessing
│   ├── feature_extraction.py         # TF-IDF feature extraction
│   ├── models.py                     # ML model implementations
│   ├── deep_learning.py              # LSTM model implementation
│   ├── evaluation.py                 # Model evaluation and visualization
│   ├── model_persistence.py          # Save/load model utilities
│   ├── threshold_optimization.py     # ROC analysis and threshold tuning
│   └── clean_space.py                # Utility functions
│
├── notebooks/
│   ├── Model.ipynb                   # Main analysis notebook
│   ├── models/saved_models/          # Saved model files and metadata
│   └── results/reports/              # Training results and reports
│
├── tests/
│   ├── test_data_loader.py           # Unit tests for data loading
│   └── test_preprocessing.py         # Unit tests for preprocessing
│
└── results/
    └── reports/                      # Model comparison reports
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Setup

1. **Clone or download this project**

2. **Create a virtual environment (recommended)**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install the package (optional)**:
```bash
pip install -e .
```

## 📊 Dataset

- **Size**: ~1,057 Khmer text samples
- **Classes**: 
  - Positive (វិជ្ជមាន)
  - Neutral (អព្យាក្រឹត)
  - Negative (អវិជ្ជមាន)
- **Source**: Social media posts and news comments in Khmer language
- **Format**: CSV with columns: `text`, `target`
- **Location**: `data/Data Collection - Sheet1.csv`

## 🚀 Usage

### 1. Training Models

Train all models with the full pipeline:
```bash
python train.py
```

Train without deep learning (faster):
```bash
python train.py --no-lstm
```

Specify custom data path:
```bash
python train.py --data_path path/to/your/data.csv
```

The training script will:
- Load and preprocess the data
- Train multiple ML models with hyperparameter optimization
- Train LSTM model (if enabled)
- Compare all models and generate reports
- Save the best model automatically
- Generate confusion matrices and ROC curves
- Save results to `results/reports/`

### 2. Making Predictions

**Command-line prediction** (single text):
```bash
python predict.py --model_path models/saved_models/best_model_*.pkl --text "ខ្ញុំចូលចិត្តផលិតផលនេះណាស់"
```

**Batch prediction** (from file):
```bash
python predict.py --model_path models/saved_models/best_model_*.pkl --input_file path/to/texts.csv
```

**With custom thresholds** (for optimized classification):
```bash
python predict.py --model_path models/saved_models/best_model_*.pkl --text "អរគុណច្រើន" --thresholds_path models/saved_models/thresholds.json
```

### 3. Running the API Server

Start the Flask API server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

**API Endpoints**:

- **GET `/`**: Home page with API documentation
- **POST `/predict`**: Predict sentiment for a single text
  ```bash
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "ខ្ញុំចូលចិត្តផលិតផលនេះណាស់"}'
  ```
  
- **POST `/predict_batch`**: Predict sentiment for multiple texts
  ```bash
  curl -X POST http://localhost:5000/predict_batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["អរគុណច្រើន", "អាក្រក់ណាស់"]}'
  ```

- **GET `/model_info`**: Get information about the loaded model
- **GET `/health`**: Health check endpoint

**Test the API**:
```bash
python test_api.py
```

### 4. Using the Jupyter Notebook

For interactive exploration and analysis:
```bash
jupyter notebook notebooks/Model.ipynb
```

The notebook includes:
- Data exploration and visualization
- Step-by-step preprocessing demonstration
- Model training and comparison
- Error analysis
- Performance visualization

## 📈 Model Performance

Based on experimental results with the Khmer dataset (~1,057 samples):

| Model | Approach | Accuracy | F1-Macro | Precision | Recall | Notes |
|-------|----------|----------|----------|-----------|--------|-------|
| **Bidirectional LSTM** | Deep Learning | **85.66%** | **85.84%** | 88.97% | 84.57% | Best overall performance |
| **SVM** | Traditional ML | 84.11% | 84.06% | 85.54% | 83.26% | Best traditional ML model |
| **Logistic Regression** | Traditional ML | 83.85% | 83.76% | 85.06% | 83.03% | Fast, interpretable |
| **XGBoost** | Gradient Boosting | 82.69% | 82.55% | 84.18% | 81.76% | Powerful for structured data |
| **Random Forest** | Ensemble | 75.97% | 75.63% | 79.24% | 74.56% | Good generalization |
| **Naive Bayes** | Probabilistic | 70.03% | 69.80% | 71.75% | 68.94% | Fast training |

*Note: Results from training on January 9, 2026. Performance metrics are from test set evaluation.*

### Performance Factors:
- **Dataset Size**: ~1,000 samples is limited for deep learning
- **Class Imbalance**: Handled via class weights
- **Preprocessing**: Khmer-specific cleaning significantly impacts results
- **Feature Engineering**: TF-IDF with n-grams provides good baseline features

## 🔍 Key Features Explained

### 1. Khmer-Specific Preprocessing

The preprocessing pipeline handles unique challenges of Khmer text:

```python
from src.preprocessing import preprocess_khmer, KHMER_SLANG

# Preprocessing steps:
# 1. Unicode normalization (NFD → NFC)
# 2. Slang dictionary mapping
# 3. Remove special characters and URLs
# 4. Preserve Khmer Unicode range (U+1780 to U+17FF)
# 5. Normalize whitespace

text = "ខ្ញុំចូលចិត្តផលិតផលនេះណាស់"
cleaned = preprocess_khmer(text, KHMER_SLANG)
```

### 2. TF-IDF Feature Extraction

Configurable TF-IDF vectorization with multiple n-gram levels:

```python
from src.feature_extraction import create_tfidf_vectorizer

# Creates vectorizer with:
# - Unigrams, bigrams, and trigrams
# - Max 5000 features
# - Min/max document frequency filtering
# - Sublinear TF scaling

vectorizer = create_tfidf_vectorizer()
```

### 3. Automated Model Training

Grid search with cross-validation for each model:

```python
from src.models import train_model_with_search

# Automatically:
# - Performs grid search
# - Uses stratified K-fold CV
# - Applies class weights
# - Saves best parameters

model = train_model_with_search(pipeline, X_train, y_train, param_grid)
```

### 4. Threshold Optimization

ROC curve analysis for optimal classification thresholds:

```python
from src.threshold_optimization import get_optimal_thresholds_multiclass

# Analyzes ROC curves for each class
# Finds optimal thresholds maximizing F1-score
# Generates visualization and recommendations

thresholds = get_optimal_thresholds_multiclass(y_true, y_proba)
```

## 📚 Module Documentation

### Core Modules

#### `src/config.py`
Configuration settings for the entire project:
- Random seeds for reproducibility
- Model parameters
- File paths
- Feature extraction settings

#### `src/data_loader.py`
Data loading and cleaning utilities:
- `load_data()`: Load CSV data
- `clean_data()`: Remove duplicates and missing values
- `prepare_train_test_split()`: Stratified train/test split
- `get_class_distribution()`: Analyze class balance

#### `src/preprocessing.py`
Khmer text preprocessing:
- `preprocess_khmer()`: Main preprocessing function
- `KHMER_SLANG`: Dictionary of informal Khmer mappings
- Unicode normalization and character filtering

#### `src/feature_extraction.py`
Feature extraction and engineering:
- `create_tfidf_vectorizer()`: Create TF-IDF vectorizer
- `compute_class_weights()`: Calculate balanced class weights

#### `src/models.py`
Machine learning model implementations:
- `create_*_pipeline()`: Create model pipelines (LR, SVM, NB, RF, XGBoost)
- `get_hyperparameter_grids()`: Get parameter grids for grid search
- `train_model_with_search()`: Train with hyperparameter optimization

#### `src/deep_learning.py`
Deep learning model implementations:
- `create_lstm_model()`: Build Bidirectional LSTM
- `prepare_sequences()`: Tokenize and pad sequences
- `train_lstm_model()`: Train LSTM with callbacks

#### `src/evaluation.py`
Model evaluation and visualization:
- `compare_models()`: Compare multiple models
- `plot_model_comparison()`: Visualize comparison
- `plot_confusion_matrix()`: Generate confusion matrices

#### `src/model_persistence.py`
Model saving and loading:
- `save_model()`: Save model with metadata
- `load_model()`: Load saved model
- `save_comparison_report()`: Save evaluation results

#### `src/threshold_optimization.py`
ROC analysis and threshold tuning:
- `compute_roc_curves_multiclass()`: Compute multi-class ROC
- `get_optimal_thresholds_multiclass()`: Find optimal thresholds
- `plot_roc_curves_multiclass()`: Visualize ROC curves
- `predict_with_threshold()`: Predict with custom thresholds

## 🔬 Khmer-Specific Challenges Addressed

1. **Unicode Complexity**: Khmer uses combining characters requiring NFD→NFC normalization
2. **Limited Resources**: Low-resource language with limited NLP tools
3. **Informal Text**: Social media contains slang and non-standard spellings
4. **Code-Switching**: Mix of Khmer and English text
5. **Class Imbalance**: Uneven distribution of sentiment classes
6. **Small Dataset**: Limited labeled data (~1,000 samples)

## 💡 Performance Improvement Strategies

### Immediate Improvements:
1. **Data Collection**: Expand dataset to 5,000-10,000+ samples
2. **Data Augmentation**: Back-translation, synonym replacement, paraphrasing
3. **Slang Dictionary**: Expand informal Khmer mappings
4. **Feature Engineering**: Add char n-grams, text length, punctuation features

### Advanced Techniques:
1. **Pre-trained Models**: Fine-tune mBERT, XLM-RoBERTa, or CamBERT
2. **Transfer Learning**: Leverage multilingual models
3. **Ensemble Methods**: Combine traditional ML + deep learning
4. **Active Learning**: Prioritize labeling of uncertain samples
5. **Cross-Validation**: K-fold validation for robust evaluation

### Production Optimization:
1. **Model Compression**: Quantization, pruning for faster inference
2. **Caching**: Cache preprocessed features
3. **Batch Processing**: Process multiple texts together
4. **API Optimization**: Async processing, load balancing

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_preprocessing.py
pytest tests/test_data_loader.py
```

## 📝 API Examples

### Python Client Example

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Single prediction
data = {"text": "ខ្ញុំចូលចិត្តផលិតផលនេះណាស់"}
response = requests.post(url, json=data)
result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function predictSentiment(text) {
  try {
    const response = await axios.post('http://localhost:5000/predict', {
      text: text
    });
    console.log('Sentiment:', response.data.sentiment);
    console.log('Confidence:', response.data.confidence);
  } catch (error) {
    console.error('Error:', error);
  }
}

predictSentiment('ខ្ញុំចូលចិត្តផលិតផលនេះណាស់');
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Data Collection**: Help label more Khmer text samples
2. **Slang Dictionary**: Expand the informal Khmer mappings
3. **Model Improvements**: Implement advanced models (BERT, etc.)
4. **Feature Engineering**: Add new features or preprocessing steps
5. **Documentation**: Improve docs and add examples
6. **Testing**: Add unit tests and integration tests

## 📄 License

This project is for educational and research purposes. Feel free to use and modify with attribution.

## 👤 Authors

**Group-02 I5-AMS-A**

Team Members:
- **Seaklong HENG**
- **Solita Chhorn**
- **Rongravidwin HAYSAVIN**
- **Ratanakvichea LONG**
- **Ratanak VITOU**

Created as part of the **I5-AMS WR Project** focusing on Khmer NLP and sentiment analysis.

## 🙏 Acknowledgments

- Khmer language resources and Unicode consortium
- scikit-learn and TensorFlow communities
- Open-source NLP research community

## 📞 Support

For questions or issues:
1. Check the documentation above
2. Review the `notebooks/Model.ipynb` for examples
3. Open an issue on the project repository

---

**Last Updated**: January 2026

**Note**: This project demonstrates practical sentiment analysis for Khmer, a low-resource language. Performance can be significantly improved with more training data and advanced transformer-based models.
