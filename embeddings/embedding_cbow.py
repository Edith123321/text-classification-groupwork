"""
Word2Vec CBOW Embedding Module for SMS Spam Classification

This module creates Word2Vec embeddings using the CBOW (Continuous Bag of Words) architecture.
CBOW is an alternative to Skip-gram with different learning dynamics.

Why CBOW:
- Learns by predicting target word from context words
- Faster to train than Skip-gram
- Works better with frequent words
- More stable on small datasets
- Good for capturing syntactic patterns

CBOW vs Skip-gram comparison:
- CBOW: Predict word from context (faster, better for common words)
- Skip-gram: Predict context from word (better for rare words)
- For SMS spam: Both should work well, CBOW trains faster

How CBOW works:
Input: Context words (e.g., ["free", "now", "limited"])
Output: Probability distribution over target word ("call")
Learning: Adjust word vectors to maximize correct word prediction
Result: Words appearing in similar contexts get similar vectors
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import pickle
import os


class CBOWEmbedder:
    """
    Creates Word2Vec CBOW embeddings for text data.
    
    Why try CBOW in addition to Skip-gram:
    - Comparison between two Word2Vec architectures
    - CBOW may perform better on this specific dataset
    - Faster training allows more hyperparameter exploration
    - Provides completeness in embedding comparison
    """
    
    def __init__(self, vector_size=100, window=5, min_count=2, sg=0, epochs=100):
        """
        Initialize CBOW parameters.
        
        Parameters explained:
        
        vector_size=100:
        - Same as Skip-gram for fair comparison
        - 100-dimensional dense vectors
        - Captures semantic meaning in compact form
        - Much smaller than TF-IDF's 5000 dimensions
        
        window=5:
        - Context window: 5 words before and after
        - CBOW averages all context words to predict target
        - Larger window = more context but slower training
        - 5 is good balance for short SMS messages
        - Example: "free [call] now" - CBOW uses "free" and "now" to predict "call"
        
        min_count=2:
        - Ignore words appearing fewer than 2 times
        - Removes noise and improves training stability
        - With 5572 messages, words appearing once don't generalize
        
        sg=0:
        - CRITICAL: sg=0 means CBOW (sg=1 would be Skip-gram)
        - This is the key difference from Skip-gram
        - CBOW: Context -> Word prediction
        - Skip-gram: Word -> Context prediction
        
        epochs=100:
        - Number of training passes over data
        - CBOW typically converges faster than Skip-gram
        - Could potentially use fewer epochs
        - 100 ensures thorough training
        
        Why CBOW might be better:
        - SMS has many repeated phrases (promotional language)
        - CBOW better captures frequent patterns
        - Faster training = more time for hyperparameter tuning
        - More stable gradients on small datasets
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg  # 0 for CBOW
        self.epochs = epochs
        self.model = None
        
    def train(self, sentences):
        """
        Train CBOW model on tokenized sentences.
        
        CBOW learning process:
        1. Take context words: ["free", "prize", "now"]
        2. Predict target word: "win"
        3. Update embeddings to improve prediction
        4. Repeat for all word positions in corpus
        
        Why train from scratch:
        - SMS language unique (abbreviations, slang)
        - Pre-trained models don't capture SMS style
        - Domain-specific embeddings outperform general ones
        - Dataset focused on spam detection task
        
        Args:
            sentences: List of tokenized sentences (list of lists)
        
        Returns:
            Trained Word2Vec CBOW model
        """
        print(f"\nTraining CBOW model on {len(sentences)} sentences...")
        print(f"Parameters:")
        print(f"  Vector size: {self.vector_size}")
        print(f"  Window size: {self.window}")
        print(f"  Min count: {self.min_count}")
        print(f"  Architecture: CBOW (sg={self.sg})")
        print(f"  Epochs: {self.epochs}")
        
        # Initialize and train model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,  # CBOW
            epochs=self.epochs,
            workers=4,  # Parallel processing for speed
            seed=42  # Reproducibility
        )
        
        print(f"\nTraining complete!")
        print(f"  Vocabulary size: {len(self.model.wv)}")
        print(f"  Vector dimensionality: {self.model.wv.vector_size}")
        
        # Compare with Skip-gram:
        # - CBOW typically trains 2-3x faster
        # - Similar vocabulary size
        # - Quality comparison requires evaluation
        
        return self.model
    
    def get_sentence_embedding(self, sentence):
        """
        Convert sentence to embedding by averaging word vectors.
        
        Why average word vectors:
        - Simple aggregation method
        - Preserves overall semantic meaning
        - Each word contributes equally
        - Works well for short texts like SMS
        
        How this creates sentence meaning:
        - "free win prize" -> Average of three spam-related vectors -> Near spam region
        - "hello how are you" -> Average of conversational vectors -> Near ham region
        - LSTM learns to classify these aggregated representations
        
        Alternative aggregation methods not used:
        - Max pooling: Takes maximum value across dimensions (loses information)
        - Weighted average by TF-IDF: More complex, marginal benefit
        - Concatenation: Too high dimensional for LSTM
        
        Args:
            sentence: List of words (tokens)
            
        Returns:
            numpy array of shape (vector_size,) representing sentence
        """
        # Collect vectors for words in vocabulary
        word_vectors = []
        for word in sentence:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        if len(word_vectors) == 0:
            # Handle out-of-vocabulary sentences
            # Return zero vector (neutral, doesn't influence prediction)
            return np.zeros(self.vector_size)
        
        # Average word vectors to get sentence embedding
        sentence_embedding = np.mean(word_vectors, axis=0)
        
        return sentence_embedding
    
    def embed_texts(self, texts):
        """
        Convert list of texts to embedding matrix.
        
        Why batch processing:
        - Efficient for large datasets
        - Returns fixed-shape array for LSTM
        - All samples have same dimensionality
        - Ready for model training
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            numpy array of shape (n_samples, vector_size)
        """
        print(f"\nEmbedding {len(texts)} texts with CBOW...")
        
        embeddings = []
        oov_count = 0  # Out of vocabulary count
        
        for text in texts:
            # Tokenize
            tokens = text.split()
            
            # Count OOV words for monitoring
            for token in tokens:
                if token not in self.model.wv:
                    oov_count += 1
            
            # Get sentence embedding
            embedding = self.get_sentence_embedding(tokens)
            embeddings.append(embedding)
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        print(f"  Embedding matrix shape: {embedding_matrix.shape}")
        print(f"  Out-of-vocabulary words: {oov_count}")
        print(f"  OOV percentage: {oov_count / sum(len(text.split()) for text in texts) * 100:.2f}%")
        
        # Why OOV matters:
        # - High OOV means many unknown words
        # - Could indicate need for lower min_count
        # - Typical OOV 5-10% is acceptable
        
        return embedding_matrix
    
    def find_similar_words(self, word, top_n=10):
        """
        Find most similar words using cosine similarity.
        
        Why analyze word similarities:
        - Validates semantic learning
        - Shows what model considers related
        - Helps understand spam word clusters
        - Useful for feature analysis
        
        Expected patterns:
        - Spam words cluster: free, win, prize, call, txt
        - Ham words cluster: hello, thanks, ok, later
        - If clusters don't make sense, investigate training
        
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
    
    def compare_words(self, word1, word2):
        """
        Compute cosine similarity between two words.
        
        Why useful:
        - Quantify semantic similarity
        - Compare spam word relationships
        - Validate embedding quality
        
        Examples to try:
        - similarity("free", "win") should be high
        - similarity("free", "hello") should be low
        - similarity("call", "txt") should be high
        
        Args:
            word1, word2: Words to compare
            
        Returns:
            Float similarity score (-1 to 1)
        """
        if word1 not in self.model.wv or word2 not in self.model.wv:
            print(f"One or both words not in vocabulary")
            return None
        
        similarity = self.model.wv.similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        
        return similarity
    
    def save_model(self, path):
        """Save trained CBOW model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"\nSaved CBOW model to {path}")
    
    @staticmethod
    def load_model(path):
        """Load saved CBOW model."""
        model = Word2Vec.load(path)
        print(f"\nLoaded CBOW model from {path}")
        embedder = CBOWEmbedder()
        embedder.model = model
        return embedder


def create_cbow_embeddings(train_df, val_df, test_df, save_dir='../data/embeddings/cbow'):
    """
    Create CBOW embeddings for all dataset splits.
    
    Complete pipeline:
    1. Train CBOW on training sentences
    2. Analyze learned embeddings
    3. Create embeddings for train/val/test
    4. Save everything for reproducibility
    
    Why separate from Skip-gram:
    - Different learning algorithm
    - May capture different patterns
    - Allows direct comparison
    - Completeness in evaluation
    
    Args:
        train_df: Training DataFrame with 'processed' column
        val_df: Validation DataFrame
        test_df: Test DataFrame
        save_dir: Directory to save embeddings and model
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, embedder)
    """
    print("="*60)
    print("CREATING CBOW EMBEDDINGS")
    print("="*60)
    
    # Create embedder with same parameters as Skip-gram for fair comparison
    embedder = CBOWEmbedder(
        vector_size=100,  # Same as Skip-gram
        window=5,         # Same as Skip-gram
        min_count=2,      # Same as Skip-gram
        sg=0,             # CBOW (key difference)
        epochs=100        # Same as Skip-gram
    )
    
    # Prepare training sentences
    print("\nPreparing training sentences...")
    train_sentences = [text.split() for text in train_df['processed'].values]
    print(f"  Prepared {len(train_sentences)} sentences")
    
    # Train CBOW model
    embedder.train(train_sentences)
    
    # Analyze learned embeddings
    print("\nAnalyzing learned embeddings...")
    
    # Check spam-related words
    spam_words = ['free', 'win', 'prize', 'call', 'txt', 'claim']
    print("\nSpam word similarities:")
    for word in spam_words:
        if word in embedder.model.wv:
            embedder.find_similar_words(word, top_n=5)
    
    # Compare word pairs
    print("\nWord pair similarities:")
    pairs = [('free', 'win'), ('free', 'hello'), ('call', 'txt'), ('prize', 'claim')]
    for word1, word2 in pairs:
        embedder.compare_words(word1, word2)
    
    # Create embeddings for all splits
    print("\nCreating embeddings for all splits...")
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
    
    embedder.save_model(f'{save_dir}/cbow_model.bin')
    
    print("\n" + "="*60)
    print("CBOW EMBEDDINGS COMPLETE")
    print("="*60)
    print(f"\nSaved to {save_dir}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    print("\nWhy CBOW embeddings:")
    print("- Faster training than Skip-gram")
    print("- Better for frequent words and phrases")
    print("- More stable on small datasets")
    print("- Comparison with Skip-gram shows which works better")
    
    print("\nExpected comparison:")
    print("- CBOW vs Skip-gram: Similar performance, CBOW faster")
    print("- Both Word2Vec vs TF-IDF: Word2Vec should win")
    print("- Semantic embeddings capture spam patterns better")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, embedder


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_pickle('../data/processed/word2vec/train.pkl')
    val_df = pd.read_pickle('../data/processed/word2vec/val.pkl')
    test_df = pd.read_pickle('../data/processed/word2vec/test.pkl')
    
    # Create CBOW embeddings
    X_train, X_val, X_test, y_train, y_val, y_test, embedder = create_cbow_embeddings(
        train_df, val_df, test_df
    )
    
    print("\nCBOW embeddings ready for LSTM training!")
