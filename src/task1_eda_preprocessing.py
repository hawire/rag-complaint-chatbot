"""
Task 1: Exploratory Data Analysis and Data Preprocessing
CFPB Financial Complaint Data Analysis for CrediTrust Financial RAG Chatbot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

def setup_paths():
    """Setup data directory paths"""
    DATA_DIR = Path('../data')
    RAW_DIR = DATA_DIR / 'raw'
    PROCESSED_DIR = DATA_DIR / 'processed'
    
    # Create directories if they don't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    return RAW_DIR, PROCESSED_DIR

def load_data(raw_dir):
    """Load the CFPB complaint dataset"""
    df = pd.read_csv(raw_dir / 'complaints.csv', low_memory=False)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def analyze_products(df):
    """Analyze product distribution and filter for target products"""
    # Show all products
    product_counts = df['Product'].value_counts()
    print("Top 10 Products by Complaint Count:")
    print(product_counts.head(10))
    
    # Target products for CrediTrust Financial
    target_products = ['Credit card', 'Personal loan', 'Savings account', 'Money transfer']
    
    # Filter for target products (case-insensitive)
    target_mask = df['Product'].str.lower().isin([p.lower() for p in target_products])
    target_df = df[target_mask].copy()
    
    print(f"\nFiltered dataset shape: {target_df.shape}")
    print("Target products distribution:")
    print(target_df['Product'].value_counts())
    
    return target_df

def analyze_narratives(target_df, narrative_col='Consumer complaint narrative'):
    """Analyze narrative content and quality"""
    # Check for missing narratives
    total_records = len(target_df)
    missing_narratives = target_df[narrative_col].isnull().sum()
    has_narratives = total_records - missing_narratives
    
    print(f"\nNarrative Analysis:")
    print(f"Total records: {total_records}")
    print(f"Records with narratives: {has_narratives} ({has_narratives/total_records*100:.1f}%)")
    print(f"Records without narratives: {missing_narratives} ({missing_narratives/total_records*100:.1f}%)")
    
    # Analyze narrative lengths
    narrative_df = target_df.dropna(subset=[narrative_col]).copy()
    narrative_df['word_count'] = narrative_df[narrative_col].str.split().str.len()
    
    print("\nNarrative length statistics:")
    print(narrative_df['word_count'].describe())
    
    # Count very short and very long narratives
    very_short = (narrative_df['word_count'] < 10).sum()
    very_long = (narrative_df['word_count'] > 1000).sum()
    print(f"Very short narratives (<10 words): {very_short}")
    print(f"Very long narratives (>1000 words): {very_long}")
    
    return narrative_df

def clean_text(text):
    """Clean and normalize complaint text"""
    if pd.isna(text):
        return text
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove common boilerplate phrases
    boilerplate_patterns = [
        r'i am writing to file a complaint',
        r'dear sir or madam',
        r'to whom it may concern',
        r'complaint department'
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(narrative_df, narrative_col='Consumer complaint narrative'):
    """Clean and preprocess the complaint data"""
    cleaned_df = narrative_df.copy()
    
    # Apply text cleaning
    cleaned_df[narrative_col] = cleaned_df[narrative_col].apply(clean_text)
    
    # Remove empty narratives after cleaning
    cleaned_df = cleaned_df[cleaned_df[narrative_col].str.len() > 10]
    
    # Standardize product categories
    product_mapping = {
        'credit card': 'Credit Card',
        'personal loan': 'Personal Loan',
        'savings account': 'Savings Account',
        'money transfer': 'Money Transfer'
    }
    
    cleaned_df['product_category'] = cleaned_df['Product'].str.lower().map(product_mapping)
    
    print(f"\nRecords after cleaning: {len(cleaned_df)}")
    print("Final product distribution:")
    print(cleaned_df['product_category'].value_counts())
    
    return cleaned_df

def save_processed_data(final_df, processed_dir, narrative_col='Consumer complaint narrative'):
    """Save the cleaned and processed dataset"""
    # Select relevant columns
    final_columns = [
        'Complaint ID', 'product_category', 'Product', 'Issue', 'Sub-issue',
        'Company', 'State', 'Date received', narrative_col
    ]
    
    # Filter columns that exist in the dataset
    available_columns = [col for col in final_columns if col in final_df.columns]
    output_df = final_df[available_columns].copy()
    
    # Save to CSV
    output_path = processed_dir / 'filtered_complaints.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Final dataset contains {len(output_df)} complaints across {output_df['product_category'].nunique()} product categories")
    
    return output_path

def main():
    """Main execution function for Task 1"""
    print("=" * 60)
    print("Task 1: EDA and Data Preprocessing")
    print("=" * 60)
    
    # Setup paths
    raw_dir, processed_dir = setup_paths()
    
    try:
        # Load data
        df = load_data(raw_dir)
        
        # Analyze and filter products
        target_df = analyze_products(df)
        
        # Analyze narratives
        narrative_df = analyze_narratives(target_df)
        
        # Preprocess data
        cleaned_df = preprocess_data(narrative_df)
        
        # Save processed data
        output_path = save_processed_data(cleaned_df, processed_dir)
        
        print("\n" + "=" * 60)
        print("Task 1 completed successfully!")
        print("=" * 60)
        
        return output_path
        
    except FileNotFoundError:
        print("\nError: Could not find complaints.csv in data/raw/")
        print("Please download the CFPB complaint dataset and place it in data/raw/complaints.csv")
        return None
    except Exception as e:
        print(f"\nError processing data: {str(e)}")
        return None

if __name__ == "__main__":
    main()