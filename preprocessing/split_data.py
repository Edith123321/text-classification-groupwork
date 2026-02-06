"""
Split cleaned data into train, validation, and test sets
"""

import random
import os

def load_cleaned_data(filepath):
    """
    Load cleaned data from TSV file
    
    Args:
        filepath: Path to cleaned data file
        
    Returns:
        List of (label, message) tuples
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return []
    
    print(f"📖 Loading cleaned data from: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            if '\t' in line:
                label, message = line.split('\t', 1)
                data.append((label, message))
            
            if line_num % 1000 == 0:
                print(f"  Loaded {line_num} messages...")
    
    print(f"✅ Loaded {len(data)} cleaned messages")
    return data

def stratified_split(data, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Perform stratified split of data by class
    
    Args:
        data: List of (label, message) tuples
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_data, val_data, test_data)
    """
    # Set random seed
    random.seed(random_seed)
    
    # Separate by class
    ham_data = [(label, msg) for label, msg in data if label.lower() == 'ham']
    spam_data = [(label, msg) for label, msg in data if label.lower() == 'spam']
    
    print(f"📊 Dataset statistics:")
    print(f"  Ham messages: {len(ham_data)}")
    print(f"  Spam messages: {len(spam_data)}")
    print(f"  Total: {len(data)}")
    
    # Shuffle each class
    random.shuffle(ham_data)
    random.shuffle(spam_data)
    
    # Calculate split sizes
    def get_split_indices(n, test_ratio, val_ratio):
        test_n = int(n * test_ratio)
        val_n = int((n - test_n) * val_ratio)
        train_n = n - test_n - val_n
        return train_n, val_n, test_n
    
    ham_train_n, ham_val_n, ham_test_n = get_split_indices(len(ham_data), test_size, val_size)
    spam_train_n, spam_val_n, spam_test_n = get_split_indices(len(spam_data), test_size, val_size)
    
    # Split each class
    ham_train = ham_data[:ham_train_n]
    ham_val = ham_data[ham_train_n:ham_train_n + ham_val_n]
    ham_test = ham_data[ham_train_n + ham_val_n:]
    
    spam_train = spam_data[:spam_train_n]
    spam_val = spam_data[spam_train_n:spam_train_n + spam_val_n]
    spam_test = spam_data[spam_train_n + spam_val_n:]
    
    # Combine
    train_data = ham_train + spam_train
    val_data = ham_val + spam_val
    test_data = ham_test + spam_test
    
    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def save_data_split(data, filepath):
    """
    Save a data split to file
    
    Args:
        data: List of (label, message) tuples
        filepath: Path to save the data
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("label\tmessage\n")
        for label, message in data:
            f.write(f"{label}\t{message}\n")
    
    print(f"  💾 Saved {len(data)} samples to {filepath}")

def main():
    """Main function"""
    print("=" * 60)
    print("DATA SPLITTING PROCESS")
    print("=" * 60)
    
    # Path to cleaned data
    cleaned_data_path = "../data/processed/cleaned_data.tsv"
    
    if not os.path.exists(cleaned_data_path):
        print(f"❌ Error: Cleaned data not found at {cleaned_data_path}")
        print("Please run clean_text.py first")
        return
    
    # Load cleaned data
    data = load_cleaned_data(cleaned_data_path)
    
    if not data:
        print("❌ No data loaded")
        return
    
    # Split the data
    print("\n✂️  Splitting data (stratified by class)...")
    train_data, val_data, test_data = stratified_split(
        data, test_size=0.2, val_size=0.1, random_seed=42
    )
    
    # Save splits
    output_dir = "../data/processed"
    
    print("\n💾 Saving data splits...")
    save_data_split(train_data, os.path.join(output_dir, "train.tsv"))
    save_data_split(val_data, os.path.join(output_dir, "val.tsv"))
    save_data_split(test_data, os.path.join(output_dir, "test.tsv"))
    
    # Print statistics
    print(f"\n✅ Data splitting complete!")
    
    print(f"\n📊 Final dataset statistics:")
    print("-" * 40)
    
    for name, split_data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
        total = len(split_data)
        ham = sum(1 for label, _ in split_data if label.lower() == 'ham')
        spam = sum(1 for label, _ in split_data if label.lower() == 'spam')
        
        print(f"{name} set:")
        print(f"  Total: {total} messages")
        print(f"  Ham: {ham} ({ham/total:.1%})")
        print(f"  Spam: {spam} ({spam/total:.1%})")
        print()

if __name__ == "__main__":
    main()