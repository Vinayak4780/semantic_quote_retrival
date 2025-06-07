from datasets import load_dataset
import pandas as pd
import os

def download_quotes():
    """
    Download quotes from the Abirate English Quotes dataset using Hugging Face datasets
    """
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists('data/raw'):
            os.makedirs('data/raw')
            
        # Load the dataset
        dataset = load_dataset("Abirate/english_quotes")
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Save the raw data
        df.to_csv('data/raw/quotes.csv', index=False)
        print(f"Downloaded {len(df)} quotes successfully!")
        return df
    
    except Exception as e:
        print(f"Error downloading quotes: {str(e)}")
        return None

if __name__ == "__main__":
    download_quotes()
