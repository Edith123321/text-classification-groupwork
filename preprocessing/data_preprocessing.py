"""
Data Preprocessing Module for SMS Spam Classification

This module handles text preprocessing with considerations for different embedding techniques.
Different embeddings require different preprocessing strategies.

Why preprocessing matters:
- Cleans noisy SMS text (URLs, special characters, numbers)
- Normalizes text for consistent embeddings
- Different embeddings benefit from different preprocessing levels
- Proper preprocessing significantly improves model performance
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pickle
import os

# Download required NLTK data (for modern NLTK 3.9+)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """
    Handles text preprocessing with embedding-specific strategies.
    
    Why different preprocessing for different embeddings:
    - TF-IDF: Benefits from stopword removal (reduces noise)
    - Word2Vec (Skip-gram/CBOW): Keep more words for context learning
    - All: Need tokenization, lowercasing, noise removal
    """
    
    def __init__(self):
        """Initialize preprocessor with stopwords and patterns."""
        self.stop_words = set(stopwords.words('english'))
        
        # Define cleaning patterns
        # Why each pattern:
        # - URLs: Not useful for spam classification, add noise
        # - Numbers: Convert to token to preserve "numerical" information
        # - Special chars: Remove but keep sentence structure
        # - Extra spaces: Normalize whitespace
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\d{3}[-.]?\d{3}[-.]?\d{4}')
        self.number_pattern = re.compile(r'\d+')
        
    def remove_urls_emails_phones(self, text):
        """
        Remove URLs, emails, and phone numbers.
        
        Why remove these:
        - URLs/emails: Vary greatly, don't generalize well
        - Phone numbers: Specific to instances, not generalizable patterns
        - These are more metadata than semantic content
        
        Alternative approach: Could replace with tokens like <URL>, <EMAIL>
        We remove completely as they're not semantic indicators for spam
        """
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = self.phone_pattern.sub('', text)
        return text
    
    def replace_numbers(self, text):
        """
        Replace numbers with <NUM> token.
        
        Why use token instead of removal:
        - Presence of numbers can indicate spam (prize amounts, phone numbers)
        - Actual number values don't matter, just that a number exists
        - Token preserves this information without vocabulary explosion
        """
        return self.number_pattern.sub('<NUM>', text)
    
    def remove_punctuation_and_special_chars(self, text):
        """
        Remove punctuation while preserving word boundaries.
        
        Why remove punctuation:
        - Reduces vocabulary size
        - "hello!" and "hello" should be same word
        - Most punctuation doesn't carry spam/ham signal
        
        Exception: We keep apostrophes in contractions (don't, isn't)
        """
        # Keep apostrophes for contractions
        text = re.sub(r"[^\w\s']", ' ', text)
        return text
    
    def to_lowercase(self, text):
        """
        Convert all text to lowercase.
        
        Why lowercase:
        - "Free" and "free" should be treated as same word
        - Reduces vocabulary size significantly
        - Makes embeddings more consistent
        - SMS often has inconsistent capitalization
        """
        return text.lower()
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Why NLTK tokenizer:
        - Handles contractions properly (don't -> do n't)
        - Better than simple split() for edge cases
        - Consistent tokenization across all samples
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove common English stopwords.
        
        When to use:
        - TF-IDF embeddings: Yes, reduces noise
        - Word2Vec embeddings: Optional, provides context
        
        Why stopwords matter:
        - TF-IDF: Stopwords have low TF-IDF scores anyway
        - Word2Vec: Stopwords provide grammatical context
        
        For this implementation: We'll make it optional
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def remove_short_tokens(self, tokens, min_length=2):
        """
        Remove very short tokens.
        
        Why remove short tokens:
        - Single letters are usually noise (typos, fragments)
        - Exception: Keep "I" and "a" which are valid words
        - Reduces vocabulary without losing information
        """
        return [token for token in tokens if len(token) >= min_length or token in ['i', 'a']]
    
    def preprocess_for_tfidf(self, text):
        """
        Preprocessing pipeline optimized for TF-IDF embeddings.
        
        Why this specific pipeline for TF-IDF:
        - TF-IDF is bag-of-words: word order doesn't matter
        - Remove stopwords: They get low TF-IDF scores anyway
        - Aggressive cleaning: Focus on discriminative words
        - Goal: Keep only words that distinguish spam from ham
        
        Steps:
        1. Clean URLs, emails, phones (noise for TF-IDF)
        2. Replace numbers with token (preserve number presence)
        3. Lowercase (normalize)
        4. Remove punctuation (focus on words)
        5. Tokenize (split into words)
        6. Remove stopwords (reduce dimensionality)
        7. Remove short tokens (reduce noise)
        """
        # Step 1-4: Clean and normalize
        text = self.remove_urls_emails_phones(text)
        text = self.replace_numbers(text)
        text = self.to_lowercase(text)
        text = self.remove_punctuation_and_special_chars(text)
        
        # Step 5: Tokenize
        tokens = self.tokenize(text)
        
        # Step 6-7: Filter tokens
        tokens = self.remove_stopwords(tokens)
        tokens = self.remove_short_tokens(tokens)
        
        # Return as space-separated string (TF-IDF expects text)
        return ' '.join(tokens)
    
    def preprocess_for_word2vec(self, text):
        """
        Preprocessing pipeline optimized for Word2Vec embeddings.
        
        Why different from TF-IDF:
        - Word2Vec learns from context: word order matters
        - Keep stopwords: They provide grammatical context
        - Less aggressive: Preserve sentence structure
        - Goal: Maintain semantic relationships while cleaning noise
        
        Why keep stopwords for Word2Vec:
        - "free call now" - "call" and "now" provide context for "free"
        - Stopwords help model learn word relationships
        - Skip-gram/CBOW use surrounding words to learn embeddings
        
        Steps:
        1. Clean URLs, emails, phones (noise)
        2. Replace numbers with token
        3. Lowercase (normalize)
        4. Remove punctuation
        5. Tokenize
        6. Keep stopwords (important for context)
        7. Remove short tokens only
        """
        # Step 1-4: Clean and normalize
        text = self.remove_urls_emails_phones(text)
        text = self.replace_numbers(text)
        text = self.to_lowercase(text)
        text = self.remove_punctuation_and_special_chars(text)
        
        # Step 5: Tokenize
        tokens = self.tokenize(text)
        
        # Step 6: Keep stopwords for context
        # Step 7: Remove only very short tokens
        tokens = self.remove_short_tokens(tokens)
        
        # Return as space-separated string
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, method='word2vec'):
        """
        Preprocess entire dataset.
        
        Args:
            df: DataFrame with 'message' and 'label' columns
            method: 'tfidf' or 'word2vec' (default: 'word2vec')
        
        Why method parameter:
        - Allows consistent preprocessing per embedding type
        - Each embedding has optimal preprocessing strategy
        - Can compare embeddings fairly with appropriate preprocessing
        
        Returns:
            DataFrame with preprocessed messages
        """
        print(f"\nPreprocessing dataset for {method} embeddings...")
        
        df_processed = df.copy()
        
        if method == 'tfidf':
            df_processed['processed'] = df['message'].apply(self.preprocess_for_tfidf)
        elif method == 'word2vec':
            df_processed['processed'] = df['message'].apply(self.preprocess_for_word2vec)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tfidf' or 'word2vec'")
        
        # Convert labels to binary (0=ham, 1=spam)
        # Why binary: LSTM output is sigmoid (0-1), need numeric labels
        df_processed['label_binary'] = (df_processed['label'] == 'spam').astype(int)
        
        print(f"Preprocessing complete. Processed {len(df_processed)} messages.")
        
        return df_processed


class DataSplitter:
    """
    Handles train/validation/test splitting with stratification.
    
    Why stratified splitting:
    - Maintains class distribution across splits
    - Critical with imbalanced data (87% ham, 13% spam)
    - Ensures validation/test sets have enough spam samples
    - Makes evaluation metrics more reliable
    """
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Initialize splitter with split ratios.
        
        Why these ratios:
        - 60% train: Enough data for LSTM to learn patterns
        - 20% validation: Monitor overfitting, tune hyperparameters
        - 20% test: Final evaluation on unseen data
        
        Why random_state=42:
        - Reproducibility: Same splits across experiments
        - Allows fair comparison between embeddings
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def split(self, df):
        """
        Split dataset into train, validation, and test sets.
        
        Why two-step splitting:
        1. First split: train+val vs test (80/20)
        2. Second split: train vs val (60/20 of original)
        
        This ensures:
        - Test set is completely held out
        - Validation set is separate from training
        - All sets maintain class balance
        """
        print("\nSplitting dataset...")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df['label_binary']  # Maintain class distribution
        )
        
        # Second split: train vs val
        # Adjust val_size to be 20% of original data
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=train_val['label_binary']
        )
        
        # Print split statistics
        print(f"\nDataset split:")
        print(f"  Training:   {len(train):4d} samples ({len(train)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val):4d} samples ({len(val)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(test):4d} samples ({len(test)/len(df)*100:.1f}%)")
        
        print(f"\nClass distribution in each set:")
        for name, dataset in [('Train', train), ('Val', val), ('Test', test)]:
            spam_pct = (dataset['label_binary'].sum() / len(dataset)) * 100
            print(f"  {name:10s}: {spam_pct:.1f}% spam")
        
        # Why this matters:
        # - All sets have similar spam percentage (~13%)
        # - No information leakage between sets
        # - Test set represents true unseen data
        
        return train, val, test
    
    def save_splits(self, train, val, test, save_dir='../data/processed'):
        """
        Save processed splits to disk.
        
        Why save to disk:
        - Avoid reprocessing for different embeddings
        - Ensures exact same splits for all experiments
        - Faster iteration during model development
        - Can reload splits without rerunning preprocessing
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as pickle files (preserves pandas DataFrame structure)
        train.to_pickle(f'{save_dir}/train.pkl')
        val.to_pickle(f'{save_dir}/val.pkl')
        test.to_pickle(f'{save_dir}/test.pkl')
        
        # Also save as CSV for human inspection
        train.to_csv(f'{save_dir}/train.csv', index=False)
        val.to_csv(f'{save_dir}/val.csv', index=False)
        test.to_csv(f'{save_dir}/test.csv', index=False)
        
        print(f"\nSaved processed splits to {save_dir}/")
        print(f"  - train.pkl, train.csv")
        print(f"  - val.pkl, val.csv")
        print(f"  - test.pkl, test.csv")


def run_preprocessing_pipeline(data_path, method='word2vec', save_dir='../data/processed'):
    """
    Run complete preprocessing pipeline.
    
    Args:
        data_path: Path to raw SMS spam collection
        method: Preprocessing method ('tfidf' or 'word2vec')
        save_dir: Directory to save processed data
    
    Why complete pipeline:
    - One function call does everything
    - Consistent preprocessing across experiments
    - Easy to modify for different embedding types
    - Reproducible results
    """
    print("="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'], encoding='utf-8')
    print(f"Loaded {len(df)} messages")
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataset(df, method=method)
    
    # Split
    splitter = DataSplitter()
    train, val, test = splitter.split(df_processed)
    
    # Save
    method_dir = f'{save_dir}/{method}'
    splitter.save_splits(train, val, test, save_dir=method_dir)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nProcessed data saved to {method_dir}/")
    print("\nNext steps:")
    print("1. Create embeddings from processed text")
    print("2. Train LSTM model with embeddings")
    print("3. Evaluate on test set")
    
    return train, val, test


if __name__ == "__main__":
    # Run preprocessing for both embedding types
    # Get the project root directory (parent of preprocessing/)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'sms+spam+collection', 'SMSSpamCollection')
    save_dir = os.path.join(project_root, 'data', 'processed')
    
    print(f"Loading data from: {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")
    print(f"Saving to: {save_dir}")
    
    print("\nPreprocessing for Word2Vec (Skip-gram and CBOW)...")
    train_w2v, val_w2v, test_w2v = run_preprocessing_pipeline(
        data_path, 
        method='word2vec',
        save_dir=save_dir
    )
    
    print("\n" + "="*60 + "\n")
    
    print("Preprocessing for TF-IDF...")
    train_tfidf, val_tfidf, test_tfidf = run_preprocessing_pipeline(
        data_path,
        method='tfidf',
        save_dir=save_dir
    )
