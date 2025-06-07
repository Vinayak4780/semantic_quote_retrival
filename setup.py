import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.download_quotes import download_quotes
from preprocess.text_processor import preprocess_quotes_data
from utils.indexer import QuoteIndexer

def main():
    # Step 1: Download data
    print("Downloading quotes...")
    df = download_quotes()
    if df is None:
        print("Failed to download quotes. Exiting...")
        return
    
    # Step 2: Preprocess data
    print("\nPreprocessing quotes...")
    processed_df = preprocess_quotes_data()
    
    # Step 3: Create FAISS index
    print("\nCreating FAISS index...")
    indexer = QuoteIndexer()
    indexer.create_index(processed_df)
    
    print("\nSetup complete! You can now run the Streamlit app using:")
    print("streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    main()
