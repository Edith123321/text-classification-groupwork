# This file contains functions for cleaning and preprocessing text data
# Removes special characters, converts to lowercase, removes stopwords, and normalizes whitespace
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('../data/sms+spam+collection/SMSSpamCollection', sep='\t', names=['label', 'message'])
    df['cleaned_text'] = df['message'].apply(clean_text)
    df.to_csv('../data/processed/cleaned_data.csv', index=False)