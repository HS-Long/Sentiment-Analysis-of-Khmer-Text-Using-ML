"""
Data preprocessing module for Khmer text sentiment analysis
"""

import re
import unicodedata
import pandas as pd


def khmer_preprocess(text):
    """
    Basic Khmer text preprocessing
    
    Args:
        text (str): Raw Khmer text
        
    Returns:
        str: Cleaned text
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Remove numbers and punctuation
    text = re.sub(r"[^\u1780-\u17FF\s]", "", text)
    
    # Remove spaces 
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def khmer_preprocess_enhanced(text, stopwords=None):
    """
    Enhanced Khmer text preprocessing with stopwords removal
    
    Args:
        text (str): Raw Khmer text
        stopwords (list, optional): List of Khmer stopwords to remove
        
    Returns:
        str: Cleaned text
    """
    if stopwords is None:
        stopwords = get_khmer_stopwords()
    
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    
    # Remove numbers and punctuation
    text = re.sub(r"[^\u1780-\u17FF\s]", "", text)
    
    # Remove spaces 
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords]
    text = " ".join(words)
    
    return text


def get_khmer_stopwords():
    """
    Returns a list of common Khmer stopwords
    
    Returns:
        list: List of Khmer stopwords
    """
    return [
        'នេះ', 'នោះ', 'នឹង', 'ដែល', 'ទៅ', 'មក', 'ឲ្យ', 'បាន', 'ជា', 'ហើយ',
        'ដើម្បី', 'ពី', 'តាម', 'រួច', 'ផង', 'ទៀត', 'ទេ', 'អត់', 'ថា', 'គឺ',
        'ក៏', 'យ៉ាង', 'របស់', 'ចំពោះ', 'ដល់', 'ជាមួយ', 'និង', 'ឬ', 'ប៉ុន្តែ'
    ]


def load_and_clean_data(filepath, use_enhanced=False):
    """
    Load and preprocess the dataset
    
    Args:
        filepath (str): Path to the CSV file
        use_enhanced (bool): Whether to use enhanced preprocessing
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Remove any rows where text == 'text' (header duplicates)
    df = df[df['text'].str.lower() != 'text']
    df.reset_index(drop=True, inplace=True)
    
    # Apply preprocessing
    if use_enhanced:
        df["clean_text"] = df["text"].apply(khmer_preprocess_enhanced)
    else:
        df["clean_text"] = df["text"].apply(khmer_preprocess)
    
    return df


def encode_labels(df, target_column='target'):
    """
    Encode sentiment labels to numeric values
    
    Args:
        df (pd.DataFrame): Dataframe with target column
        target_column (str): Name of the target column
        
    Returns:
        tuple: (df with encoded labels, LabelEncoder object)
    """
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[target_column])
    
    return df, label_encoder
