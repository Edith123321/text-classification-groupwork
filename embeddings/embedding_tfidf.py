"""
TF-IDF Embedding Module for SMS Spam Classification

This module creates TF-IDF (Term Frequency-Inverse Document Frequency) embeddings.
TF-IDF is a statistical baseline that doesn't capture semantic relationships.

Why TF-IDF as baseline:
- Simple and fast to compute
- No training required (unlike Word2Vec)
- Provides interpretable feature importance
- Shows benefit of semantic embeddings by comparison
- Industry standard for text classification baselines

How TF-IDF works:
- TF: Term Frequency - how often word appears in document
- IDF: Inverse Document Frequency - how unique word is across corpus
- TF-IDF = TF * IDF - high values for important unique words

Example:
- Common word "the": High TF, low IDF -> Low TF-IDF (not important)
- Spam word "prize": Medium TF, high IDF -> High TF-IDF (important)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


class TFIDFEmbedder:
    """
    Creates TF-IDF embeddings for text data.
    
    Why use TF-IDF for LSTM:
    - Provides statistical baseline to compare against Word2Vec
    - Fast computation for quick iterations
    - Shows that semantic embeddings (Word2Vec) are better
    - TF-IDF features are more interpretable than Word2Vec
    """
    
    def __init__(self, max_features=5000, max_df=0.95, min_df=2):
        """
        Initialize TF-IDF vectorizer with parameters.
        
        Parameters explained:
        
        max_features=5000:
        - Why: Limits vocabulary to 5000 most important words
        - Benefit: Reduces dimensionality, faster training
        - Trade-off: Might miss some rare but important spam words
        - Chosen because: SMS corpus has ~8000 unique words, 5000 captures most information
        
        max_df=0.95:
        - Why: Ignore words appearing in >95% of documents
        - Benefit: Removes extremely common words (the, is, to)
        - These words don't distinguish spam from ham
        - Alternative to stopword removal
        
        min_df=2:
        - Why: Ignore words appearing in <2 documents
        - Benefit: Removes typos, rare words that don't generalize
        - Trade-off: Might remove some unique spam indicators
        - Chosen because: With 5572 messages, words appearing once are likely noise
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,  # Apply sublinear tf scaling (log) - reduces impact of very frequent words
            strip_accents='unicode',  # Remove accents
            analyzer='word',  # Tokenize by words
            token_pattern=r'\w{1,}',  # Words of 1+ characters
            ngram_range=(1, 2),  # Use unigrams and bigrams
            use_idf=True  # Use inverse-document-frequency reweighting
        )
        
        # Why ngram_range=(1, 2):
        # - Unigrams: Individual words like "free", "prize"
        # - Bigrams: Word pairs like "call_now", "click_here"
        # - Spam often uses specific phrases, not just individual words
        # - Trade-off: Increases feature space but captures more patterns
        
        self.feature_names = None
        
    def fit_transform(self, texts):
        """
        Fit vectorizer on training data and transform.
        
        Why fit only on training data:
        - Prevents information leakage from validation/test sets
        - Models what would happen with truly unseen data
        - If we fit on all data, we'd be "cheating" by knowing test vocabulary
        
        Args:
            texts: List or array of preprocessed text strings
            
        Returns:
            Dense numpy array of TF-IDF features (samples x max_features)
        """
        print(f"\nFitting TF-IDF vectorizer on {len(texts)} training samples...")
        
        # Fit and transform training data
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Store feature names for interpretability
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"  Vocabulary size: {len(self.feature_names)}")
        print(f"  Feature matrix shape: {tfidf_matrix.shape}")
        print(f"  Matrix sparsity: {(1 - tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)) * 100:.2f}%")
        
        # Why sparsity matters:
        # - TF-IDF matrices are very sparse (most values are 0)
        # - A document only contains a small subset of vocabulary
        # - For LSTM, we need dense input, so we'll convert
        
        # Convert sparse matrix to dense for LSTM
        # Why dense for LSTM:
        # - LSTM expects dense input tensors
        # - Sparse matrices save memory but LSTM can't use them directly
        # - With max_features=5000, density is manageable
        dense_matrix = tfidf_matrix.toarray()
        
        print(f"  Converted to dense matrix: {dense_matrix.shape}")
        
        return dense_matrix
    
    def transform(self, texts):
        """
        Transform new data using fitted vectorizer.
        
        Why separate transform:
        - Use same vocabulary learned from training
        - Validation and test sets might have new words
        - New words are ignored (not in vocabulary)
        - This simulates real-world deployment
        
        Args:
            texts: List or array of preprocessed text strings
            
        Returns:
            Dense numpy array of TF-IDF features
        """
        print(f"\nTransforming {len(texts)} samples with trained TF-IDF vectorizer...")
        
        tfidf_matrix = self.vectorizer.transform(texts)
        dense_matrix = tfidf_matrix.toarray()
        
        print(f"  Transformed matrix shape: {dense_matrix.shape}")
        
        return dense_matrix
    
    def get_top_features_per_class(self, X, y, top_n=20):
        """
        Get top TF-IDF features for each class.
        
        Why analyze top features:
        - Understand what words distinguish spam from ham
        - Validate that features make intuitive sense
        - Identify potential data quality issues
        - Helps explain model predictions
        
        Args:
            X: TF-IDF feature matrix
            y: Binary labels (0=ham, 1=spam)
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        print(f"\nAnalyzing top {top_n} features per class...")
        
        # Compute mean TF-IDF score for each feature in each class
        ham_indices = y == 0
        spam_indices = y == 1
        
        ham_mean = X[ham_indices].mean(axis=0)
        spam_mean = X[spam_indices].mean(axis=0)
        
        # Get top features
        ham_top_indices = ham_mean.argsort()[-top_n:][::-1]
        spam_top_indices = spam_mean.argsort()[-top_n:][::-1]
        
        ham_top_features = [(self.feature_names[i], ham_mean[i]) for i in ham_top_indices]
        spam_top_features = [(self.feature_names[i], spam_mean[i]) for i in spam_top_indices]
        
        print(f"\nTop {top_n} features for HAM:")
        for feature, score in ham_top_features:
            print(f"  {feature:20s}: {score:.4f}")
        
        print(f"\nTop {top_n} features for SPAM:")
        for feature, score in spam_top_features:
            print(f"  {feature:20s}: {score:.4f}")
        
        # Why this is useful:
        # - Spam features should include: "free", "call", "prize", "txt", "win"
        # - Ham features should be: normal conversation words
        # - If features don't make sense, check preprocessing
        
        return {
            'ham': ham_top_features,
            'spam': spam_top_features
        }
    
    def save_vectorizer(self, path):
        """
        Save fitted vectorizer for later use.
        
        Why save:
        - Can reuse for inference without retraining
        - Ensures consistent feature extraction
        - Required for deployment
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"\nSaved TF-IDF vectorizer to {path}")
    
    @staticmethod
    def load_vectorizer(path):
        """Load saved vectorizer."""
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"\nLoaded TF-IDF vectorizer from {path}")
        return vectorizer


def create_tfidf_embeddings(train_df, val_df, test_df, save_dir='../data/embeddings/tfidf'):
    """
    Create TF-IDF embeddings for all dataset splits.
    
    Why separate function:
    - Encapsulates entire TF-IDF pipeline
    - Easy to call from main experiment script
    - Handles all data loading and saving
    - Ensures consistent processing
    
    Args:
        train_df: Training DataFrame with 'processed' column
        val_df: Validation DataFrame
        test_df: Test DataFrame
        save_dir: Directory to save embeddings and vectorizer
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, embedder)
    """
    print("="*60)
    print("CREATING TF-IDF EMBEDDINGS")
    print("="*60)
    
    # Create embedder
    embedder = TFIDFEmbedder(max_features=5000, max_df=0.95, min_df=2)
    
    # Fit on training data and transform
    X_train = embedder.fit_transform(train_df['processed'].values)
    y_train = train_df['label_binary'].values
    
    # Transform validation and test data
    X_val = embedder.transform(val_df['processed'].values)
    y_val = val_df['label_binary'].values
    
    X_test = embedder.transform(test_df['processed'].values)
    y_test = test_df['label_binary'].values
    
    # Analyze top features
    top_features = embedder.get_top_features_per_class(X_train, y_train, top_n=15)
    
    # Save embeddings and vectorizer
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f'{save_dir}/X_train.npy', X_train)
    np.save(f'{save_dir}/X_val.npy', X_val)
    np.save(f'{save_dir}/X_test.npy', X_test)
    np.save(f'{save_dir}/y_train.npy', y_train)
    np.save(f'{save_dir}/y_val.npy', y_val)
    np.save(f'{save_dir}/y_test.npy', y_test)
    
    embedder.save_vectorizer(f'{save_dir}/tfidf_vectorizer.pkl')
    
    print("\n" + "="*60)
    print("TF-IDF EMBEDDINGS COMPLETE")
    print("="*60)
    print(f"\nSaved to {save_dir}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    print("\nWhy TF-IDF features:")
    print("- Fast baseline for comparison")
    print("- Shows which words are statistically important")
    print("- No semantic understanding (just frequency)")
    print("- Expected to underperform Word2Vec embeddings")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, embedder


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_pickle('../data/processed/tfidf/train.pkl')
    val_df = pd.read_pickle('../data/processed/tfidf/val.pkl')
    test_df = pd.read_pickle('../data/processed/tfidf/test.pkl')
    
    # Create TF-IDF embeddings
    X_train, X_val, X_test, y_train, y_val, y_test, embedder = create_tfidf_embeddings(
        train_df, val_df, test_df
    )
    
    print("\nTF-IDF embeddings ready for LSTM training!")
