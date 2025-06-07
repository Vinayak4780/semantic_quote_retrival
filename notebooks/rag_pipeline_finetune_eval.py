"""
RAG Pipeline: Fine-tuning, LLM Integration, and Evaluation (Python Script)
- Fine-tune a sentence-transformer on the quotes dataset
- Use the fine-tuned model for semantic retrieval
- Integrate Groq Llama-2 for answer/summary generation
- Evaluate the RAG pipeline with RAGAS
"""

import os
import random
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss
import numpy as np
import requests
from ragas.metrics import answer_relevancy, context_precision, context_recall
from ragas import evaluate

# 1. Load and preprocess the dataset
ds = load_dataset("Abirate/english_quotes")
df = pd.DataFrame(ds['train'])
df = df.dropna(subset=['quote', 'author', 'tags'])
df['quote'] = df['quote'].str.lower().str.strip()
df['author'] = df['author'].str.lower().str.strip()
df['tags'] = df['tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
df = df[df['quote'] != '']
df = df.reset_index(drop=True)

# 2. Fine-tune a SentenceTransformer on the Quotes Dataset
examples = []
for i, row in df.iterrows():
    context = f"{row['author']} | {row['tags']}"
    examples.append(InputExample(texts=[row['quote'], context]))
random.shuffle(examples)
train_examples = examples[:int(0.9*len(examples))]
val_examples = examples[int(0.9*len(examples)) :]
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, batch_size=32, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,  # Increase for better results
    warmup_steps=100,
    show_progress_bar=True
)
model.save('finetuned-quote-sbert')

# 3. Build FAISS Index with Fine-tuned Model
embeddings = model.encode(df['quote'].tolist(), show_progress_bar=True, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, 'faiss_index/quotes_finetuned.faiss')
df.to_pickle('faiss_index/quotes_finetuned.pkl')

# 4. Integrate Groq Llama-2 for Answer/Summary Generation
def generate_llama2_summary_groq(query, retrieved_quotes, groq_api_key):
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

# 5. RAG Evaluation with RAGAS
def ragas_evaluate(query, retrieved, llm_json):
    results = evaluate(
        queries=[query],
        answers=[llm_json['summary']],
        contexts=[[q['quote'] for q in retrieved]],
        metrics=[answer_relevancy, context_precision, context_recall]
    )
    return results

# Example usage:
# query = "quotes about hope by oscar wilde"
# retrieved = [
#     {"quote": "We are all in the gutter, but some of us are looking at the stars.", "author": "oscar wilde", "tags": "hope, stars"},
# ]
# llm_json = generate_llama2_summary_groq(query, retrieved, os.environ["GROQ_API_KEY"])
# print(ragas_evaluate(query, retrieved, llm_json))
