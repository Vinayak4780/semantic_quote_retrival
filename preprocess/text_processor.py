import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import pickle
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess text by performing the following steps:
        1. Convert to lowercase
        2. Remove special characters and numbers
        3. Remove stopwords
        4. Remove extra whitespace
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join tokens back together
        text = ' '.join(tokens)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

def preprocess_quotes_data(input_path='data/raw/quotes.csv', output_path='data/processed'):
    """
    Preprocess the quotes dataset and save processed data
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess quotes
    df['processed_quote'] = df['quote'].apply(preprocessor.preprocess_text)
    
    # Remove rows with empty processed quotes
    df = df[df['processed_quote'].str.strip() != '']
    
    # Save processed data
    df.to_csv(f'{output_path}/processed_quotes.csv', index=False)
    
    # Save preprocessor for later use
    with open(f'{output_path}/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Processed {len(df)} quotes successfully!")
    return df

if __name__ == "__main__":
    preprocess_quotes_data()
