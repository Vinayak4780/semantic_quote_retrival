import streamlit as st
import sys
import os
import json
import pickle
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.text_processor import TextPreprocessor
from utils.indexer import QuoteIndexer
from evaluation.evaluator import RAGEvaluator

# Initialize components
@st.cache_resource
def load_components():
    # Load preprocessor
    with open('data/processed/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load indexer
    indexer = QuoteIndexer()
    indexer.load_index()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    return preprocessor, indexer, evaluator

def generate_llama2_summary_groq(query, retrieved_quotes, groq_api_key):
    import requests
    context = '\n'.join([f"Quote: {q['quote']}\nAuthor: {q['author']}\nTags: {q['tags']}" for q in retrieved_quotes])
    prompt = f"""
You are a helpful assistant. Given the following user query and a set of quotes, return a structured JSON with:
- relevant quotes
- authors
- tags
- a short summary answering the query

User Query: {query}
Quotes:
{context}

Return JSON with keys: quotes, authors, tags, summary.
"""
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": "llama2-70b-4096", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    return response.json()['choices'][0]['message']['content']

def main():
    st.title("🎯 Quote Retrieval System")
    st.write("Enter a query to find relevant quotes!")
    
    try:
        preprocessor, indexer, evaluator = load_components()
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        return
    
    # Query input
    query = st.text_input("Enter your query:", placeholder="Enter a query about hope...")
    
    if query:
        # Preprocess query
        processed_query = preprocessor.preprocess_text(query)
        
        # Search quotes
        results = indexer.search(processed_query, k=5)
        
        # Evaluate results
        evaluation_metrics = evaluator.evaluate_retrieval(processed_query, results)
        
        # LLM summary and structured JSON
        groq_api_key = os.getenv('GROQ_API_KEY', '')
        if groq_api_key:
            llm_json = generate_llama2_summary_groq(query, results, groq_api_key)
            st.subheader('🧠 LLM Structured Response')
            st.json(llm_json)
        
        # Display results
        st.subheader("📊 Retrieved Quotes")
        for i, result in enumerate(results, 1):
            with st.expander(f"Quote {i} (Similarity: {result['similarity_score']:.3f})"):
                st.markdown(f"**Quote:** _{result['quote']}_")
                st.markdown(f"**Author:** {result['author']}")
                st.markdown(f"**Tags:** {result['tags']}")
        
        # Display evaluation metrics
        st.subheader("📈 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg. Similarity", f"{evaluation_metrics['average_similarity']:.3f}")
        
        with col2:
            st.metric("Max Similarity", f"{evaluation_metrics['max_similarity']:.3f}")
        
        with col3:
            st.metric("Diversity Score", f"{evaluation_metrics['diversity_score']:.3f}")

if __name__ == "__main__":
    main()
