# RAG-Based Semantic Quote Retrieval System

## Overview
This project implements a full Retrieval Augmented Generation (RAG) pipeline for semantic quote retrieval using the [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes) dataset. It covers data preparation, model fine-tuning, vector search with FAISS, LLM-based answer generation (Groq Llama-2), RAG evaluation, and a user-friendly Streamlit app.

---

## Features
- **Data Preparation:** Download, clean, and preprocess quotes (tokenization, lowercasing, stopword removal, etc.).
- **Model Fine-Tuning:** Fine-tune a sentence-transformer (MiniLM) on the quotes for semantic search.
- **Vector Indexing:** Build a FAISS index for fast quote retrieval.
- **LLM Integration:** Use Groq Llama-2 to generate structured answers and summaries from retrieved quotes.
- **RAG Evaluation:** Evaluate the pipeline using custom and RAGAS metrics.
- **Streamlit App:** Interactive UI for querying, retrieval, LLM answers, and evaluation metrics.

---

## Project Structure
```
├── data/
│   ├── download_quotes.py         # Download and save raw dataset
│   ├── raw/                       # Raw CSV data
│   └── processed/                 # Cleaned CSV and preprocessor
├── preprocess/
│   └── text_processor.py          # Data cleaning and preprocessing
├── finetuned-quote-sbert/         # Fine-tuned model checkpoint and config
├── faiss_index/
│   ├── quotes_finetuned.faiss     # FAISS index for retrieval
│   ├── quotes_finetuned.pkl       # Quote metadata
│   ├── quotes.faiss               # (Legacy/alt index)
│   └── quotes_data.pkl            # (Legacy/alt metadata)
├── notebooks/
│   └── rag_pipeline_finetune_eval.py # End-to-end RAG pipeline (train, index, LLM, eval)
├── evaluation/
│   └── evaluator.py               # Custom evaluation metrics
├── streamlit_app/
│   └── app.py                     # Streamlit UI for querying and results
├── utils/
│   └── indexer.py                 # FAISS index handling
├── requirements.txt               # All dependencies
├── setup.py                       # Script for data and index setup
└── README.md                      # This file
```

---

## How It Works
1. **Data Preparation:**
   - Run `data/download_quotes.py` to fetch the dataset.
   - Run `preprocess/text_processor.py` to clean and preprocess the data.

2. **Model Fine-Tuning & Indexing:**
   - Run `notebooks/rag_pipeline_finetune_eval.py` to:
     - Fine-tune the sentence-transformer on the quotes.
     - Build and save the FAISS index and metadata.

3. **Streamlit App:**
   - Run `streamlit run streamlit_app/app.py`.
   - Enter a query (e.g., "quotes about hope by Oscar Wilde").
   - The app retrieves relevant quotes, calls Groq Llama-2 for a structured answer, and displays results and evaluation metrics.

4. **RAG Evaluation:**
   - Use the evaluation functions in `notebooks/rag_pipeline_finetune_eval.py` or `evaluation/evaluator.py` to assess retrieval and answer quality.

---

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies
- Groq API key (for Llama-2 integration)

---

## Usage
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and preprocess data
python data/download_quotes.py
python preprocess/text_processor.py

# 3. Fine-tune model and build index
python notebooks/rag_pipeline_finetune_eval.py

# 4. Run the Streamlit app
streamlit run streamlit_app/app.py
```

---

## Configuration
- Fine-tuned model and config are saved in `finetuned-quote-sbert/`.
- FAISS index and metadata are in `faiss_index/`.
- Set your Groq API key as the environment variable `GROQ_API_KEY` for LLM features.

---

## Credits
- Dataset: [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
- Sentence Transformers, FAISS, HuggingFace, Groq, RAGAS, Streamlit

---

## License
This project is for educational and research purposes only.
#
