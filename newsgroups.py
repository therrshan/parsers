"""
20 Newsgroups Dataset Parser

This script parses the 20 Newsgroups dataset from raw files using the provided parsing logic.
Assumes the raw files are already downloaded and stored in a 'data' folder.
NLTK is required for text processing features.

Features:
- Category filtering
- Text cleaning & preprocessing options
- Vectorization (TF-IDF, Count)
- Stop word removal
- Return as tokens or cleaned text
- Multiple return formats (numpy, dataframe)

Usage:
    from newsgroups_parser import parse_newsgroups
    
    # Get the full dataset as numpy arrays
    texts, targets = parse_newsgroups(return_type='numpy')
    
    # Get only specific categories with TF-IDF vectorization
    X, y = parse_newsgroups(
        categories=['alt.atheism', 'comp.graphics'],
        vectorize='tfidf',
        return_type='numpy'
    )
    
    # Get clean text with stop words removed
    texts, targets = parse_newsgroups(
        clean_text=True,
        remove_stopwords=True,
        return_type='numpy'
    )
    
    # Get tokenized text
    tokens, targets = parse_newsgroups(
        return_tokens=True,
        return_type='numpy'
    )
    
    # Get the dataset as a pandas DataFrame with Count vectorization
    df = parse_newsgroups(
        vectorize='count',
        return_type='dataframe'
    )
"""

import os
import re
import numpy as np
import pandas as pd
from os.path import join
import random
from typing import List, Union, Tuple, Dict, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    sklearn_available = True
except ImportError:
    sklearn_available = False


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    text = re.sub(r'From:.*?\n', '', text)
    text = re.sub(r'Subject:.*?\n', '', text)

    text = re.sub(r'[\n\r]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text.lower())

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


def tokenize_text(text: str, remove_stopwords: bool = False) -> List[str]:
    text = re.sub(r'From:.*?\n', '', text)
    text = re.sub(r'Subject:.*?\n', '', text)
    
    text = re.sub(r'[\n\r]', ' ', text)
    tokens = word_tokenize(text.lower())
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    return tokens


def parse_newsgroups(
    data_dir: str = './data/20NG',
    categories: List[str] = None,
    clean_text: bool = False,
    remove_stopwords: bool = False,
    return_tokens: bool = False,
    vectorize: str = None,
    max_features: int = 10000,
    split: bool = False,
    train_ratio: float = 0.8,
    random_state: int = 42,
    return_type: str = 'numpy'
):

    random.seed(random_state)
    np.random.seed(random_state)

    if vectorize is not None and vectorize not in ['tfidf', 'count']:
        raise ValueError(f"Invalid vectorize value: {vectorize}. Choose from 'tfidf', 'count', or None.")
    
    if vectorize and not sklearn_available:
        raise ImportError("scikit-learn is required for text vectorization.")

    raw_data = []
    target = []

    train_path = join(data_dir, "20news-bydate-train")
    test_path = join(data_dir, "20news-bydate-test")

    datasets = [train_path, test_path]
    dataset_labels = ['train', 'test'] 
    split_info = []

    for dataset_idx, dataset in enumerate(datasets):
        for category in os.listdir(dataset):
            if categories is not None and category not in categories:
                continue
                
            category_path = os.path.join(dataset, category)
            if os.path.isdir(category_path):
                for document in os.listdir(category_path):
                    document_path = os.path.join(category_path, document)
                    with open(document_path, "r", errors="ignore") as f:
                        text = f.read()
                        raw_data.append(text)
                    target.append(category)
                    split_info.append(dataset_labels[dataset_idx])

    if clean_text or return_tokens or vectorize:
        processed_data = []
        
        for text in raw_data:
            if return_tokens:
                tokens = tokenize_text(text, remove_stopwords)
                processed_data.append(tokens)
            elif clean_text:
                cleaned = clean_text(text, remove_stopwords)
                processed_data.append(cleaned)
            else:
                processed_data.append(text)
    else:
        processed_data = raw_data

    if return_tokens:
        df = pd.DataFrame({
            'raw_text': raw_data, 
            'tokens': [' '.join(tokens) for tokens in processed_data],
            'target': target, 
            'split': split_info
        })
    else:
        df = pd.DataFrame({
            'text': processed_data, 
            'target': target, 
            'split': split_info
        })

        if clean_text:
            df['raw_text'] = raw_data

    if vectorize:
        if vectorize == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features)
        elif vectorize == 'count':
            vectorizer = CountVectorizer(max_features=max_features)

        text_col = 'tokens' if return_tokens else 'text'
        X = vectorizer.fit_transform(df[text_col])

        feature_names = vectorizer.get_feature_names_out()
        df_vectorized = pd.DataFrame(X.toarray(), columns=feature_names)

        df_vectorized['target'] = df['target']
        df_vectorized['split'] = df['split']

        if 'raw_text' in df.columns:
            df_vectorized['raw_text'] = df['raw_text']
        
        df = df_vectorized
    
    if split and train_ratio != 0.6:  
        indices = np.random.permutation(len(df))
        df = df.iloc[indices].reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        df.loc[:split_idx, 'split'] = 'train'
        df.loc[split_idx:, 'split'] = 'test'
    
    if return_type == 'dataframe':
        return df
    
    elif return_type == 'numpy':
        if split:
            train_df = df[df['split'] == 'train']
            test_df = df[df['split'] == 'test']
            
            if vectorize:
                feature_cols = [col for col in df.columns if col not in ['target', 'split', 'raw_text']]
                
                return (
                    train_df[feature_cols].values,
                    test_df[feature_cols].values,
                    train_df['target'].values,
                    test_df['target'].values
                )
            elif return_tokens:
                return (
                    train_df['tokens'].apply(lambda x: x.split()).values,
                    test_df['tokens'].apply(lambda x: x.split()).values,
                    train_df['target'].values,
                    test_df['target'].values
                )
            else:
                text_col = 'text' if 'text' in train_df.columns else 'raw_text'
                return (
                    train_df[text_col].values,
                    test_df[text_col].values,
                    train_df['target'].values,
                    test_df['target'].values
                )
        else:
            if vectorize:
                feature_cols = [col for col in df.columns if col not in ['target', 'split', 'raw_text']]
                return df[feature_cols].values, df['target'].values
            elif return_tokens:
                return df['tokens'].apply(lambda x: x.split()).values, df['target'].values
            else:
                text_col = 'text' if 'text' in df.columns else 'raw_text'
                return df[text_col].values, df['target'].values
    
    else:
        raise ValueError(f"Unknown return type: {return_type}")


if __name__ == '__main__':

    df = parse_newsgroups(return_type='dataframe')
    print(f"Full dataset shape: {df.shape}")
    print(f"Categories: {df['target'].nunique()}")

    df_clean = parse_newsgroups(clean_text=True, remove_stopwords=True, return_type='dataframe')
    print(f"Clean text example: {df_clean['text'].iloc[0][:100]}")

    df_tokens = parse_newsgroups(return_tokens=True, return_type='dataframe')
    print(f"Tokens example: {df_tokens['tokens'].iloc[0][:100]}")

    df_tfidf = parse_newsgroups(vectorize='tfidf', max_features=1000, return_type='dataframe')
    print(f"TF-IDF DataFrame shape: {df_tfidf.shape}")

    df_count = parse_newsgroups(vectorize='count', max_features=1000, return_type='dataframe')
    print(f"Count DataFrame shape: {df_count.shape}")

    selected_categories = ['alt.atheism', 'comp.graphics', 'sci.space']
    df_filtered = parse_newsgroups(
        categories=selected_categories, 
        vectorize='tfidf',
        return_type='dataframe'
    )
    print(f"Filtered vectorized dataset shape: {df_filtered.shape}")