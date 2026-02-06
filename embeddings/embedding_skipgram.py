
"""
Word2Vec Skip-gram Embedding Module for SMS Spam Classification

This module creates Word2Vec embeddings using the Skip-gram architecture.
Unlike TF-IDF, Word2Vec captures semantic relationships between words.

Why Skip-gram:
- Learns word embeddings by predicting context words from target word
- Captures semantic meaning: "free" and "win" have similar vectors
- Better than TF-IDF for understanding spam patterns
- Works well with small to medium datasets
- Good for rare words (learns from being in different contexts)

Skip-gram vs CBOW:
- Skip-gram: Predict context from word (better for rare words)
- CBOW: Predict word from context (faster, better for frequent words)
- Skip-gram chosen because: SMS has many important rare spam words

How Skip-gram works:
Input: Target word (e.g., "prize")
Output: Probability distribution over context words
Learning: Adjust word vectors to maximize correct context prediction
Result: Words in similar contexts get similar vectors
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import pickle
import os


class SkipgramEmbedder:
    """
    Creates Word2Vec Skip-gram embeddings for text data.
    
    Why Word2Vec for LSTM:
    - Captures semantic relationships that TF-IDF misses
    - Dense low-dimensional vectors (100-300d) vs sparse high-dim TF-IDF (5000d)
    - Similar spam words get similar embeddings
    - LSTM can learn from semantic patterns
    """
    
    def __init__(self, vector_size=100, window=5, min_count=2, sg=1, epochs=100):
        """
        Initialize Skip-gram parameters.
        
        Parameters explained:
        
        vector_size=100:
        - Why: Dimensionality of word embeddings
        - Trade-off: Larger = more information, but more parameters to train
        - 100-300 is standard; 100 chosen for faster training
        - Each word becomes a 100-dimensional vector
        - Benefit: Much smaller than TF-IDF's 5000 dimensions
        
        window=5:
        - Why: Context window size (words before and after target)
        - window=5 means look at 5 words on each side
        - Trade-off: Larger window = more context but slower training
        - SMS messages are short (avg 15 words), so 5 is reasonable
        - Example: "free call now win prize" - "free" sees ["call", "now", "win", "prize"]
        
        min_count=2:
        - Why: Ignore words appearing fewer than 2 times
        - Benefit: Removes typos and rare words that don't generalize
        - With 5572 messages, words appearing once are likely noise
        - Trade-off: Might lose some unique spam indicators
        
        sg=1:
        - Why: Use Skip-gram (sg=1) instead of CBOW (sg=0)
        - Skip-gram better for: rare words, small datasets, semantic similarity
        - CBOW better for: frequent words, very large datasets, speed
        - Chosen because: SMS has important rare spam words
        
        epochs=100:
        - Why: Number of training iterations over corpus
        - More epochs = better embeddings but longer training
        - 100 is reasonable for small corpus (5572 messages)
        - Early stopping if convergence detected
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.epochs = epochs
        self.model = None
        
    def train(self, sentences):
        """
        Train Skip-gram model on tokenized sentences.
        
        Why train from scratch vs pre-trained:
        - SMS language is unique (abbreviations, slang)
        - Pre-trained models (Google News, Wikipedia) don't capture SMS style
        - Domain-specific embeddings perform better
        - Small dataset but focused vocabulary
        
        Args:
            sentences: List of tokenized sentences (list of lists)
                      Example: [['free', 'call', 'now'], ['hello', 'how', 'are', 'you']]
        
        Returns:
            Trained Word2Vec model
        """
        print(f"\nTraining Skip-gram model on {len(sentences)} sentences...")
        print(f"Parameters:")
        print(f"  Vector size: {self.vector_size}")
        print(f"  Window size: {self.window}")
        print(f"  Min count: {self.min_count}")
        print(f"  Epochs: {self.epochs}")
        
        # Initialize and train model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,  # Skip-gram
            epochs=self.epochs,
            workers=4,  # Parallel processing
            seed=42  # Reproducibility
        )
        
        print(f"\nTraining complete!")
        print(f"  Vocabulary size: {len(self.model.wv)}")
        print(f"  Vector dimensionality: {self.model.wv.vector_size}")
        
        # Why vocabulary size matters:
        # - Shows how many unique words model learned
        # - Should be ~3000-5000 for SMS corpus (after min_count filtering)
        # - Smaller than TF-IDF vocabulary but captures semantic meaning
        
        return self.model
    
    def get_sentence_embedding(self, sentence):
        """
        Convert sentence to embedding by averaging word vectors.
        
        Why average:
        - Simple and effective aggregation method
        - Preserves semantic meaning of sentence
        - Each word contributes equally to sentence meaning
        - Alternative: Weighted average by TF-IDF (more complex)
        
        Why this works:
        - "free call now" averages to vector near spam region
        - "hello how are you" averages to vector near ham region
        - LSTM can learn patterns in these sentence embeddings
        
        Args:
            sentence: List of words (tokens)
            
        Returns:
            numpy array of shape (vector_size,) representing sentence
        """
        # Get vectors for words in vocabulary
        word_vectors = []
        for word in sentence:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if len(word_vectors) == 0:
            # If no words in vocabulary, return zero vector
            # Why: Out-of-vocabulary sentences need some representation
            # Zero vector is neutral (doesn't push prediction either way)
            return np.zeros(self.vector_size)
        
        # Average word vectors
        # Why mean: Treats all words equally important
        # Alternative: Could weight by word importance
        sentence_embedding = np.mean(word_vectors, axis=0)
        
        return sentence_embedding
    
    def embed_texts(self, texts):
        """
        Convert list of texts to embedding matrix.
        
        Why batch processing:
        - Efficient for large datasets
        - Returns numpy array ready for LSTM
        - Consistent shape across all samples
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            numpy array of shape (n_samples, vector_size)
        """
        print(f"\nEmbedding {len(texts)} texts...")
        
        embeddings = []
        for text in texts:
            # Tokenize (split by whitespace)
            tokens = text.split()
            
            # Get sentence embedding
            embedding = self.get_sentence_embedding(tokens)
            embeddings.append(embedding)
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        print(f"  Embedding matrix shape: {embedding_matrix.shape}")
        print(f"  Expected shape: ({len(texts)}, {self.vector_size})")
        
        return embedding_matrix
    
    def find_similar_words(self, word, top_n=10):
        """
        Find most similar words to given word.
        
        Why useful:
        - Validates that embeddings capture semantic meaning
        - Helps understand what model learned
        - Can identify spam word clusters
        - Useful for feature engineering
        
        Args:
            word: Word to find similar words for
            top_n: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if word not in self.model.wv:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        similar_words = self.model.wv.most_similar(word, topn=top_n)
        
        print(f"\nMost similar words to '{word}':")
        for similar_word, score in similar_words:
            print(f"  {similar_word:20s}: {score:.4f}")
        
        # Why this matters:
        # - If "free" is similar to "win", "prize", "call" -> Good embeddings!
        # - If similarities don't make sense -> Check preprocessing or training
        # - Semantic clusters help LSTM recognize spam patterns
        
        return similar_words
    
    def get_similar_words(self, word, topn=10):
        """Alias for find_similar_words for notebook compatibility."""
        if word not in self.model.wv:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self.model.wv.most_similar(word, topn=topn)
    
    def compare_embeddings_with(self, other_embedder, word):
        """
        Compare this embedder's word embedding with another embedder.
        
        Args:
            other_embedder: Another Word2Vec embedder (Skip-gram or CBOW)
            word: Word to compare
            
        Returns:
            Dictionary with comparison results
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if word not in self.model.wv or word not in other_embedder.model.wv:
            return None
        
        vec1 = self.model.wv[word]
        vec2 = other_embedder.model.wv[word]
        
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        
        return {
            'word': word,
            'similarity': similarity,
            'vec1_sample': vec1[:10],
            'vec2_sample': vec2[:10]
        }
    
    def save_model(self, path):
        """
        Save trained Word2Vec model.
        
        Why save:
        - Reuse embeddings without retraining
        - Consistent embeddings across experiments
        - Required for deployment
        - Training takes time, don't want to repeat
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"\nSaved Skip-gram model to {path}")
    
    @staticmethod
    def load_model(path):
        """Load saved Word2Vec model."""
        model = Word2Vec.load(path)
        print(f"\nLoaded Skip-gram model from {path}")
        embedder = SkipgramEmbedder()
        embedder.model = model
        return embedder


def create_skipgram_embeddings(train_df, val_df, test_df, save_dir='../data/embeddings/skipgram'):
    """
    Create Skip-gram embeddings for all dataset splits.
    
    Why this pipeline:
    - Train on training data only (prevent leakage)
    - Apply same embeddings to val/test
    - Save everything for reproducibility
    - One function call for complete workflow
    
    Args:
        train_df: Training DataFrame with 'processed' column
        val_df: Validation DataFrame
        test_df: Test DataFrame
        save_dir: Directory to save embeddings and model
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, embedder)
    """
    print("="*60)
    print("CREATING SKIP-GRAM EMBEDDINGS")
    print("="*60)
    
    # Create embedder
    embedder = SkipgramEmbedder(
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,  # Skip-gram
        epochs=100
    )
    
    # Prepare training sentences (list of token lists)
    print("\nPreparing training sentences...")
    train_sentences = [text.split() for text in train_df['processed'].values]
    print(f"  Prepared {len(train_sentences)} sentences")
    
    # Train Skip-gram model
    embedder.train(train_sentences)
    
    # Analyze some example words
    print("\nAnalyzing learned embeddings...")
    spam_words = ['free', 'win', 'prize', 'call', 'txt']
    for word in spam_words:
        if word in embedder.model.wv:
            embedder.find_similar_words(word, top_n=5)
    
    # Create embeddings for all splits
    X_train = embedder.embed_texts(train_df['processed'].values)
    y_train = train_df['label_binary'].values
    
    X_val = embedder.embed_texts(val_df['processed'].values)
    y_val = val_df['label_binary'].values
    
    X_test = embedder.embed_texts(test_df['processed'].values)
    y_test = test_df['label_binary'].values
    
    # Save embeddings and model
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f'{save_dir}/X_train.npy', X_train)
    np.save(f'{save_dir}/X_val.npy', X_val)
    np.save(f'{save_dir}/X_test.npy', X_test)
    np.save(f'{save_dir}/y_train.npy', y_train)
    np.save(f'{save_dir}/y_val.npy', y_val)
    np.save(f'{save_dir}/y_test.npy', y_test)
    
    embedder.save_model(f'{save_dir}/skipgram_model.bin')
    
    print("\n" + "="*60)
    print("SKIP-GRAM EMBEDDINGS COMPLETE")
    print("="*60)
    print(f"\nSaved to {save_dir}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    print("\nWhy Skip-gram embeddings:")
    print("- Captures semantic word relationships")
    print("- Dense 100-dim vectors vs sparse 5000-dim TF-IDF")
    print("- Similar spam words get similar embeddings")
    print("- Expected to outperform TF-IDF baseline")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, embedder


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_pickle('../data/processed/word2vec/train.pkl')
    val_df = pd.read_pickle('../data/processed/word2vec/val.pkl')
    test_df = pd.read_pickle('../data/processed/word2vec/test.pkl')
    
    # Create Skip-gram embeddings
    X_train, X_val, X_test, y_train, y_val, y_test, embedder = create_skipgram_embeddings(
        train_df, val_df, test_df
    )
    
    print("\nSkip-gram embeddings ready for LSTM training!")
