# This file handles loading raw data and splitting it into train, validation, and test sets
# Maps labels and saves processed data to CSV files
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""
    df = pd.read_csv('../data/processed/cleaned_data.csv')
    
    # First split: train+val and test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    
    # Second split: train and val
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), 
        stratify=train_val['label'], random_state=random_state
    )
    
    train.to_csv('../data/processed/train.csv', index=False)
    val.to_csv('../data/processed/val.csv', index=False)
    test.to_csv('../data/processed/test.csv', index=False)
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

if __name__ == "__main__":
    split_data()