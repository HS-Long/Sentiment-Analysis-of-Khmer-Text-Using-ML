"""
Feature extraction module for Khmer sentiment analysis
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def create_tfidf_vectorizer(ngram_range=(1, 2), max_features=3000, 
                            min_df=1, max_df=1.0, sublinear_tf=False):
    """
    Create a TF-IDF vectorizer with specified parameters
    
    Args:
        ngram_range (tuple): Range of n-grams to extract
        max_features (int): Maximum number of features
        min_df (int or float): Minimum document frequency
        max_df (float): Maximum document frequency
        sublinear_tf (bool): Apply sublinear tf scaling
        
    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )
    
    return vectorizer


def create_enhanced_vectorizer():
    """
    Create an enhanced TF-IDF vectorizer with optimized parameters
    
    Returns:
        TfidfVectorizer: Enhanced vectorizer
    """
    return create_tfidf_vectorizer(
        ngram_range=(1, 3),      # unigrams + bigrams + trigrams
        max_features=5000,       # increased features
        min_df=2,                # ignore rare terms
        max_df=0.8,              # ignore very common terms
        sublinear_tf=True        # apply sublinear tf scaling
    )


def extract_features(df, text_column='clean_text', vectorizer=None, 
                    fit=True, enhanced=False):
    """
    Extract TF-IDF features from text
    
    Args:
        df (pd.DataFrame): Dataframe with text column
        text_column (str): Name of the text column
        vectorizer (TfidfVectorizer, optional): Pre-fitted vectorizer
        fit (bool): Whether to fit the vectorizer
        enhanced (bool): Use enhanced vectorizer settings
        
    Returns:
        tuple: (feature matrix, vectorizer)
    """
    if vectorizer is None:
        if enhanced:
            vectorizer = create_enhanced_vectorizer()
        else:
            vectorizer = create_tfidf_vectorizer()
    
    if fit:
        X = vectorizer.fit_transform(df[text_column])
    else:
        X = vectorizer.transform(df[text_column])
    
    return X, vectorizer


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into train and test sets
    
    Args:
        X: Feature matrix
        y: Labels
        test_size (float): Proportion of test set
        random_state (int): Random seed
        stratify (bool): Whether to stratify split by labels
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test
