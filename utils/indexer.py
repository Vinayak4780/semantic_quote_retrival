import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

class QuoteIndexer:
    def __init__(self, model_name='finetuned-quote-sbert'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.quotes_data = None
        
    def create_index(self, quotes_df, output_dir='faiss_index'):
        """
        Create FAISS index from processed quotes
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(quotes_df['processed_quote'].tolist(), show_progress_bar=True)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        # Save the index
        faiss.write_index(self.index, f'{output_dir}/quotes.faiss')
        
        # Save the quotes data
        self.quotes_data = quotes_df
        with open(f'{output_dir}/quotes_data.pkl', 'wb') as f:
            pickle.dump(quotes_df, f)
            
        print("Index created and saved successfully!")
        
    def load_index(self, index_path='faiss_index/quotes.faiss', data_path='faiss_index/quotes_data.pkl'):
        """
        Load existing FAISS index and quotes data
        """
        self.index = faiss.read_index(index_path)
        with open(data_path, 'rb') as f:
            self.quotes_data = pickle.load(f)
            
    def search(self, query, k=5):
        """
        Search for similar quotes
        """
        # Encode query
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        
        # Search
        D, I = self.index.search(query_vector, k)
        
        # Get results
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            quote_data = self.quotes_data.iloc[idx]
            results.append({
                'quote': quote_data['quote'],
                'author': quote_data['author'],
                'tags': quote_data['tags'],
                'similarity_score': float(dist)
            })
        
        return results

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/processed_quotes.csv')
    
    # Create and save index
    indexer = QuoteIndexer()
    indexer.create_index(df)
