"""
Text tokenization utilities
Note: Named tokenize.py but uses _tokenize_text() internally to avoid conflicts
"""

import re
import numpy as np
from collections import Counter
import pickle
import os
import string

class TextTokenizer:
    """
    Custom text tokenizer for NLP tasks
    Uses _tokenize_text() internally to avoid conflict with Python's tokenize module
    """
    
    def __init__(self, max_vocab_size=10000, min_freq=1, 
                 oov_token='<UNK>', pad_token='<PAD>'):
        """
        Initialize tokenizer
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include
            oov_token: Token for out-of-vocabulary words
            pad_token: Token for padding
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.oov_token = oov_token
        self.pad_token = pad_token
        
        # Vocabulary mappings
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.word_freq = Counter()
        
        # Special tokens (add these first)
        self.special_tokens = {
            'pad': pad_token,
            'unk': oov_token,
            'bos': '<BOS>',  # Beginning of sequence
            'eos': '<EOS>'   # End of sequence
        }
        
        # Punctuation to remove
        self.punctuation = string.punctuation + '0123456789'
    
    def _tokenize_text(self, text, remove_punct=True):
        """
        Tokenize a single text string
        Named _tokenize_text to avoid conflict with Python's tokenize module
        
        Args:
            text: Input text
            remove_punct: Whether to remove punctuation
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation if specified
        if remove_punct:
            # Remove punctuation characters
            text = text.translate(str.maketrans('', '', self.punctuation))
        
        # Split on whitespace
        tokens = text.split()
        
        # Clean tokens
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text strings
        """
        print("🔨 Building vocabulary...")
        
        # Reset counters
        self.word_freq = Counter()
        
        # Count word frequencies
        for text in texts:
            tokens = self._tokenize_text(text)
            self.word_freq.update(tokens)
        
        # Filter words by frequency
        valid_words = []
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.special_tokens.values():
                valid_words.append((word, freq))
        
        # Sort by frequency (descending)
        valid_words.sort(key=lambda x: (-x[1], x[0]))
        
        # Limit vocabulary size (reserve space for special tokens)
        max_regular = self.max_vocab_size - len(self.special_tokens)
        if len(valid_words) > max_regular:
            valid_words = valid_words[:max_regular]
        
        # Build index mappings
        idx = 0
        
        # Add special tokens first
        for token_name, token in self.special_tokens.items():
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1
        
        # Add regular words
        for word, _ in valid_words:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1
        
        self.vocab_size = len(self.word2idx)
        
        print(f"✅ Vocabulary built:")
        print(f"   Size: {self.vocab_size} tokens")
        print(f"   Regular words: {len(valid_words)}")
        
        if valid_words:
            top_words = [word for word, _ in valid_words[:10]]
            print(f"   Top 10 words: {top_words}")
    
    def texts_to_sequences(self, texts, max_length=None):
        """
        Convert texts to sequences of indices
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length (None for variable length)
            
        Returns:
            Numpy array of sequences
        """
        sequences = []
        
        for text in texts:
            tokens = self._tokenize_text(text)
            seq = []
            
            for token in tokens:
                if token in self.word2idx:
                    seq.append(self.word2idx[token])
                else:
                    seq.append(self.word2idx[self.oov_token])
            
            # Pad or truncate if max_length specified
            if max_length is not None:
                if len(seq) < max_length:
                    # Pad
                    seq = seq + [self.word2idx[self.pad_token]] * (max_length - len(seq))
                else:
                    # Truncate
                    seq = seq[:max_length]
            
            sequences.append(seq)
        
        return np.array(sequences)
    
    def sequences_to_texts(self, sequences):
        """
        Convert sequences back to text
        
        Args:
            sequences: Numpy array of sequences
            
        Returns:
            List of text strings
        """
        texts = []
        
        for seq in sequences:
            tokens = []
            for idx in seq:
                if idx in self.idx2word:
                    word = self.idx2word[idx]
                    # Skip special tokens in output
                    if word not in self.special_tokens.values():
                        tokens.append(word)
            
            texts.append(' '.join(tokens))
        
        return texts
    
    def save(self, filepath):
        """
        Save tokenizer to file
        
        Args:
            filepath: Path to save file
        """
        # Create directory if needed
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'max_vocab_size': self.max_vocab_size,
                'min_freq': self.min_freq,
                'oov_token': self.oov_token,
                'pad_token': self.pad_token,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'word_freq': dict(self.word_freq),
                'special_tokens': self.special_tokens
            }, f)
        
        print(f"✅ Tokenizer saved to {filepath}")
    
    def load(self, filepath):
        """
        Load tokenizer from file
        
        Args:
            filepath: Path to load file from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.max_vocab_size = data['max_vocab_size']
            self.min_freq = data['min_freq']
            self.oov_token = data['oov_token']
            self.pad_token = data['pad_token']
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
            self.word_freq = Counter(data['word_freq'])
            self.special_tokens = data['special_tokens']
            
            print(f"✅ Tokenizer loaded from {filepath}")
            print(f"   Vocabulary size: {self.vocab_size}")
            
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise

# Convenience functions
def tokenize_simple(text, lowercase=True, remove_punct=True):
    """
    Simple tokenization function
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_punct: Remove punctuation
        
    Returns:
        List of tokens
    """
    if not isinstance(text, str):
        return []
    
    if lowercase:
        text = text.lower()
    
    if remove_punct:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
    
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens

def test_tokenizer():
    """Test the tokenizer"""
    print("=" * 60)
    print("TOKENIZER TEST")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "Hello world! This is a test.",
        "Another example for testing tokenization.",
        "Hello world appears again here."
    ]
    
    # Create tokenizer
    tokenizer = TextTokenizer(max_vocab_size=50, min_freq=1)
    tokenizer.build_vocabulary(texts)
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(texts, max_length=10)
    
    print(f"\n🔢 Sample sequences (max_length=10):")
    for i, seq in enumerate(sequences):
        print(f"  Text {i+1}: {seq}")
    
    # Convert back to text
    reconstructed = tokenizer.sequences_to_texts(sequences)
    
    print(f"\n📝 Reconstructed texts:")
    for i, text in enumerate(reconstructed):
        print(f"  Text {i+1}: {text}")
    
    # Test save/load
    print(f"\n💾 Testing save/load...")
    tokenizer.save('test_tokenizer.pkl')
    
    # Load it back
    new_tokenizer = TextTokenizer()
    new_tokenizer.load('test_tokenizer.pkl')
    
    # Clean up
    if os.path.exists('test_tokenizer.pkl'):
        os.remove('test_tokenizer.pkl')
        print("✅ Cleaned up test file")
    
    print("\n✅ Tokenizer test complete!")

if __name__ == "__main__":
    test_tokenizer()