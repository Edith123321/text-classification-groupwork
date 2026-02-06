#!/usr/bin/python3
"""
CBOWEmbedding module

Implements a reusable CBOW (Continuous Bag of Words) embedding
wrapper that integrates cleanly with the main training pipeline.

Designed to be used as:
    from embeddings.embedding_cbow import CBOWEmbedding
"""

import numpy as np
from gensim.models import Word2Vec


class CBOWEmbedder:
    """
    CBOWEmbedding

    Wrapper around gensim Word2Vec (CBOW mode) that exposes:
    - train(all_texts)
    - get_embedding_matrix(word_index)

    This interface matches the expected usage in the main model code.
    """

    def __init__(self, vector_size=100, window=5, min_count=2, seed=42):
        """
        Initialize CBOW embedding parameters.

        Args:
            vector_size (int): Dimensionality of word vectors
            window (int): Context window size
            min_count (int): Minimum word frequency
            seed (int): Random seed for reproducibility
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.model = None

    # --------------------------------------------------

    def train(self, texts):
        """
        Train CBOW Word2Vec model on all available text.

        Args:
            texts (list[str]): List of preprocessed text strings
        """
        print("Training CBOW Word2Vec embeddings...")

        sentences = [text.split() for text in texts if isinstance(text, str)]

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=0,           
            workers=4,
            seed=self.seed
        )

        print(f"CBOW training complete")
        print(f"Vocabulary size: {len(self.model.wv)}")
        print(f"Embedding dimension: {self.vector_size}")

    # --------------------------------------------------

    def get_embedding_matrix(self, word_index):
        """
        Create embedding matrix aligned with tokenizer word_index.

        This is designed for Keras / PyTorch Embedding layers.

        Args:
            word_index (dict): Mapping {word: index}

        Returns:
            np.ndarray: Embedding matrix of shape
                        (vocab_size + 1, vector_size)
        """
        if self.model is None:
            raise ValueError("CBOW model has not been trained yet")

        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.vector_size))

        for word, idx in word_index.items():
            if word in self.model.wv:
                embedding_matrix[idx] = self.model.wv[word]
            else:
                # Random init for OOV words
                embedding_matrix[idx] = np.random.normal(
                    scale=0.1, size=(self.vector_size,)
                )

        return embedding_matrix
