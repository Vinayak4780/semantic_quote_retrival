import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class RAGEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def calculate_semantic_similarity(self, query: str, retrieved_quotes: List[Dict]) -> Dict:
        """
        Calculate semantic similarity between query and retrieved quotes
        """
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Encode retrieved quotes
        retrieved_texts = [item['quote'] for item in retrieved_quotes]
        quote_embeddings = self.model.encode(retrieved_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, quote_embeddings)[0]
        
        # Calculate metrics
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        return {
            'average_similarity': float(avg_similarity),
            'max_similarity': float(max_similarity),
            'individual_similarities': similarities.tolist()
        }
    
    def evaluate_diversity(self, retrieved_quotes: List[Dict]) -> Dict:
        """
        Evaluate diversity of retrieved quotes
        """
        # Encode retrieved quotes
        retrieved_texts = [item['quote'] for item in retrieved_quotes]
        quote_embeddings = self.model.encode(retrieved_texts)
        
        # Calculate pairwise similarities
        pairwise_similarities = cosine_similarity(quote_embeddings)
        
        # Calculate diversity metrics
        avg_pairwise_similarity = (np.sum(pairwise_similarities) - len(retrieved_quotes)) / (len(retrieved_quotes) * (len(retrieved_quotes) - 1))
        
        return {
            'average_pairwise_similarity': float(avg_pairwise_similarity),
            'diversity_score': float(1 - avg_pairwise_similarity)
        }
    
    def evaluate_retrieval(self, query: str, retrieved_quotes: List[Dict]) -> Dict:
        """
        Evaluate retrieval results
        """
        semantic_metrics = self.calculate_semantic_similarity(query, retrieved_quotes)
        diversity_metrics = self.evaluate_diversity(retrieved_quotes)
        
        return {
            **semantic_metrics,
            **diversity_metrics,
            'num_results': len(retrieved_quotes)
        }

if __name__ == "__main__":
    # Test evaluation
    evaluator = RAGEvaluator()
    # Add test cases here
