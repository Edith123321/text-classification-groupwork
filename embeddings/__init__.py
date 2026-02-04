"""Embeddings module for SMS Spam Classification"""
from .embedding_tfidf import TFIDFEmbedder
from .embedding_skipgram import SkipgramEmbedder
from .embedding_cbow import CBOWEmbedder

__all__ = ['TFIDFEmbedder', 'SkipgramEmbedder', 'CBOWEmbedder']
