"""
Data Exploration Module for SMS Spam Classification

This module performs comprehensive exploratory data analysis (EDA) on the SMS spam dataset.
The goal is to understand data characteristics before preprocessing and modeling.

Why this approach:
- Understanding class distribution helps us address imbalance issues
- Text length analysis informs sequence padding strategies for LSTM
- Vocabulary analysis guides embedding dimension choices
- Statistical insights support preprocessing decisions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os


class DataExplorer:
    """
    Performs exploratory data analysis on SMS spam dataset.
    
    Why we need this:
    - Identifies class imbalance (affects training strategy)
    - Reveals text characteristics (affects LSTM architecture)
    - Guides preprocessing decisions based on data patterns
    """
    
    def __init__(self, data_path):
        """
        Initialize explorer with dataset path.
        
        Args:
            data_path: Path to SMS spam collection file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """
        Load SMS spam dataset from tab-separated file.
        
        Why this format:
        - Original SMS Spam Collection uses tab-separated format
        - Two columns: label (ham/spam) and message text
        - No header row in the original file
        """
        print("Loading SMS Spam Collection dataset...")
        
        # Load with explicit column names since no header
        self.df = pd.read_csv(
            self.data_path,
            sep='\t',
            header=None,
            names=['label', 'message'],
            encoding='utf-8'
        )
        
        print(f"Loaded {len(self.df)} messages")
        return self.df
    
    def basic_statistics(self):
        """
        Compute and display basic dataset statistics.
        
        Why these metrics matter:
        - Total samples: Determines if we have enough data for deep learning
        - Class distribution: Reveals imbalance that requires stratified splitting
        - Class percentages: Shows minority class proportion (affects evaluation metrics)
        """
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        total = len(self.df)
        ham_count = (self.df['label'] == 'ham').sum()
        spam_count = (self.df['label'] == 'spam').sum()
        
        print(f"Total messages: {total}")
        print(f"Ham (legitimate) messages: {ham_count} ({ham_count/total*100:.2f}%)")
        print(f"Spam messages: {spam_count} ({spam_count/total*100:.2f}%)")
        print(f"Class imbalance ratio: {ham_count/spam_count:.2f}:1")
        
        # Why this matters: Heavily imbalanced dataset (87% ham, 13% spam)
        # Implications: Need stratified splits, F1-score more important than accuracy
        
        return {
            'total': total,
            'ham': ham_count,
            'spam': spam_count,
            'imbalance_ratio': ham_count/spam_count
        }
    
    def text_length_analysis(self):
        """
        Analyze message length distributions.
        
        Why length matters for LSTM:
        - Determines max sequence length for padding
        - Short sequences are efficient for LSTM processing
        - Extreme outliers may need truncation
        - Average length guides LSTM hidden units choice
        """
        print("\n" + "="*60)
        print("TEXT LENGTH ANALYSIS")
        print("="*60)
        
        # Character length
        self.df['char_length'] = self.df['message'].str.len()
        
        # Word count (simple whitespace split)
        self.df['word_count'] = self.df['message'].str.split().str.len()
        
        print("\nCharacter Length Statistics:")
        print(self.df['char_length'].describe())
        
        print("\nWord Count Statistics:")
        print(self.df['word_count'].describe())
        
        print("\nLength by Class:")
        print(self.df.groupby('label')[['char_length', 'word_count']].mean())
        
        # Why this is important:
        # - SMS messages are short (avg ~80 chars, ~15 words)
        # - LSTM can handle variable length well
        # - Spam tends to be slightly longer (more promotional text)
        # - We'll pad sequences to max length for batch processing
        
        return self.df[['char_length', 'word_count']]
    
    def vocabulary_analysis(self):
        """
        Analyze vocabulary characteristics.
        
        Why vocabulary analysis matters:
        - Total unique words determines embedding vocabulary size
        - Word frequency distribution guides min_df parameter for TF-IDF
        - Rare words might be noise or important spam indicators
        - Vocabulary size affects model memory requirements
        """
        print("\n" + "="*60)
        print("VOCABULARY ANALYSIS")
        print("="*60)
        
        # Tokenize all messages (simple split for analysis)
        all_words = []
        for message in self.df['message']:
            words = message.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        word_freq = Counter(all_words)
        
        print(f"Total words (with repetition): {len(all_words)}")
        print(f"Unique words (vocabulary size): {len(word_freq)}")
        print(f"Average words per message: {len(all_words)/len(self.df):.2f}")
        
        print("\nMost common words (top 20):")
        for word, count in word_freq.most_common(20):
            print(f"  {word}: {count}")
        
        # Why this matters:
        # - ~8000 unique words in vocabulary
        # - Most common words are stopwords (the, to, i, you)
        # - For TF-IDF: we'll limit to top 5000 features
        # - For Word2Vec: we'll use all words with min_count=2
        
        return word_freq
    
    def class_distribution_visualization(self, save_dir=None):
        """
        Create visualizations of class distribution.
        
        Why visualize:
        - Visual confirmation of class imbalance
        - Helps communicate dataset characteristics in reports
        - Identifies if stratified sampling is necessary (yes, it is)
        """
        if save_dir is None:
            # Use absolute path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            save_dir = os.path.join(project_root, 'results', 'figures')
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class count bar plot
        class_counts = self.df['label'].value_counts()
        axes[0].bar(class_counts.index, class_counts.values, color=['green', 'red'])
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_ylim(0, max(class_counts.values) * 1.1)
        
        # Add count labels on bars
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontsize=12)
        
        # Class percentage pie chart
        axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                   colors=['green', 'red'], startangle=90)
        axes[1].set_title('Class Percentage Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved class distribution plot to {save_dir}/class_distribution.png")
        plt.close()
        
    def text_length_visualization(self, save_dir=None):
        """
        Create visualizations of text length distributions.
        
        Why this matters for LSTM:
        - Shows distribution of sequence lengths we'll process
        - Identifies appropriate max_length for padding
        - Reveals if truncation is needed for outliers
        """
        if save_dir is None:
            # Use absolute path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            save_dir = os.path.join(project_root, 'results', 'figures')
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Word count distribution
        axes[0, 0].hist(self.df['word_count'], bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Words')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['word_count'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["word_count"].mean():.1f}')
        axes[0, 0].legend()
        
        # Word count by class
        ham_words = self.df[self.df['label'] == 'ham']['word_count']
        spam_words = self.df[self.df['label'] == 'spam']['word_count']
        axes[0, 1].hist([ham_words, spam_words], bins=30, label=['Ham', 'Spam'],
                       color=['green', 'red'], alpha=0.6)
        axes[0, 1].set_title('Word Count by Class', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Character length distribution
        axes[1, 0].hist(self.df['char_length'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Character Length Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Characters')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(self.df['char_length'].mean(), color='red', linestyle='--',
                          label=f'Mean: {self.df["char_length"].mean():.1f}')
        axes[1, 0].legend()
        
        # Box plot comparison
        data_to_plot = [ham_words, spam_words]
        axes[1, 1].boxplot(data_to_plot, labels=['Ham', 'Spam'])
        axes[1, 1].set_title('Word Count Comparison (Box Plot)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Words')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/text_length_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved text length analysis to {save_dir}/text_length_analysis.png")
        plt.close()
    
    def run_full_analysis(self):
        """
        Execute complete exploratory data analysis pipeline.
        
        Why we do this before modeling:
        - Understand data before making preprocessing decisions
        - Identify issues that need addressing (imbalance, outliers)
        - Guide hyperparameter choices based on data characteristics
        - Create baseline understanding for result interpretation
        """
        print("Starting comprehensive data exploration...")
        
        # Load data
        self.load_data()
        
        # Compute statistics
        stats = self.basic_statistics()
        lengths = self.text_length_analysis()
        vocab = self.vocabulary_analysis()
        
        # Create visualizations
        self.class_distribution_visualization()
        self.text_length_visualization()
        
        print("\n" + "="*60)
        print("EXPLORATION COMPLETE")
        print("="*60)
        print("\nKey Findings:")
        print(f"1. Dataset has {stats['imbalance_ratio']:.2f}:1 class imbalance")
        print(f"2. Average message length: {self.df['word_count'].mean():.1f} words")
        print(f"3. Vocabulary size: ~{len(vocab):,} unique words")
        print("\nImplications for LSTM model:")
        print("- Use stratified train/val/test split")
        print("- F1-score will be primary evaluation metric")
        print("- Pad sequences to handle variable length")
        print("- Consider class weights or oversampling for minority class")
        
        return self.df


if __name__ == "__main__":
    # Run exploration
    # Get the project root directory (parent of preprocessing/)
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'sms+spam+collection', 'SMSSpamCollection')
    
    print(f"Loading data from: {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")
    
    explorer = DataExplorer(data_path)
    df = explorer.run_full_analysis()
