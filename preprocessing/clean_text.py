"""
Text cleaning for SMS spam dataset
Cleans text by removing special characters, converting to lowercase, etc.
"""

import re
import os
from pathlib import Path

def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_clean_dataset():
    """
    Load the SMS spam dataset and clean all messages
    
    Returns:
        List of (label, cleaned_message) tuples
    """
    # Try different possible paths
    possible_paths = [
        "../data/sms+spam+collection/SMSSpamCollection",
        "data/sms+spam+collection/SMSSpamCollection",
        "SMSSpamCollection",
        "../SMSSpamCollection"
    ]
    
    input_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path = path
            print(f"📁 Found data at: {path}")
            break
    
    if not input_path:
        print("❌ Error: Could not find SMSSpamCollection file")
        return []
    
    cleaned_data = []
    
    print("🧹 Cleaning text data...")
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Split by tab (label\tmessage format)
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    label, message = parts
                    cleaned = clean_text(message)
                    cleaned_data.append((label, cleaned))
            
            # Show progress
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} messages...")
    
    print(f"✅ Cleaned {len(cleaned_data)} messages")
    return cleaned_data

def save_cleaned_data(cleaned_data, output_path):
    """
    Save cleaned data to file
    
    Args:
        cleaned_data: List of (label, message) tuples
        output_path: Path to save the data
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("label\tmessage\n")
        for label, message in cleaned_data:
            f.write(f"{label}\t{message}\n")
    
    print(f"💾 Saved cleaned data to: {output_path}")

def main():
    """Main function"""
    print("=" * 60)
    print("TEXT CLEANING PROCESS")
    print("=" * 60)
    
    # Load and clean data
    cleaned_data = load_and_clean_dataset()
    
    if not cleaned_data:
        print("❌ No data to save")
        return
    
    # Save cleaned data
    output_path = "../data/processed/cleaned_data.tsv"
    save_cleaned_data(cleaned_data, output_path)
    
    # Show samples
    print("\n📝 Sample cleaned messages:")
    for i, (label, message) in enumerate(cleaned_data[:5]):
        print(f"{i+1}. Label: {label}")
        print(f"   Message: {message[:60]}..." if len(message) > 60 else f"   Message: {message}")
        print()

if __name__ == "__main__":
    main()