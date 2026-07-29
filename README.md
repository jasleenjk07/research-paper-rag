# 📚 Research Paper RAG Assistant

> A production-style **Retrieval-Augmented Generation (RAG)** system for answering questions across multiple Machine Learning and NLP research papers using **Hybrid Retrieval (Dense + BM25)**, **Cross-Encoder Reranking**, and a **local Large Language Model (Qwen via Ollama)** with citation-grounded responses.

---

## ✨ Features

- 📄 Ingest and index multiple research papers (PDF)
- ✂️ Intelligent document chunking
- 🔎 Dense semantic retrieval using **BGE Small**
- 📚 Lexical retrieval using **BM25**
- 🔀 Hybrid retrieval using **Reciprocal Rank Fusion (RRF)**
- 🎯 Cross-Encoder reranking for improved relevance
- 🤖 Local inference using **Qwen 3.5** through **Ollama**
- 📖 Citation-grounded answers with page references
- 📊 Retrieval evaluation using Recall@K and MRR
- 📝 Answer evaluation dataset with ground-truth responses
- 🌐 Interactive Streamlit dashboard
- ⚡ Fully local pipeline (no paid API required)

---

# Demo

## Ask Questions

- Ask natural language questions about indexed research papers.
- Receive grounded answers with supporting citations.

## Retrieved Sources

Each answer includes:

- Research paper title
- Page number
- Chunk ID
- Retrieval score
- Retrieved context

## Evaluation Dashboard

Browse:

- Generated answers
- Ground truth
- Retrieved contexts
- Citations
- Category-wise questions

---

# System Architecture

```text
                        Research Papers (PDFs)
                                  │
                                  ▼
                           PDF Loader
                                  │
                                  ▼
                      Recursive Text Chunking
                                  │
                                  ▼
                     BGE Small Embedding Model
                                  │
                                  ▼
                            FAISS Index
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             Dense Retrieval               BM25 Retrieval
                    └─────────────┬─────────────┘
                                  ▼
                    Reciprocal Rank Fusion (RRF)
                                  ▼
                     Cross-Encoder Reranker
                                  ▼
                      Top-K Relevant Chunks
                                  ▼
                     Qwen 3.5 (Ollama Local)
                                  ▼
              Citation-Grounded Final Response
```

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| UI | Streamlit |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS |
| Lexical Retriever | BM25 |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Qwen 3.5 (Ollama) |
| Evaluation | Custom Evaluation + Retrieval Metrics |

---

# Project Structure

```text
research-paper-rag/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── papers/
│
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── llm.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   └── ingest.py
│
├── evaluation/
│   ├── questions.json
│   ├── results.json
│   ├── run_evaluation.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_bm25.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_reranker.py
│   └── evaluate_answers.py
│
└── storage/
    ├── faiss.index
    ├── chunks.pkl
    └── processed_papers.json
```

---

# Retrieval Pipeline

The retrieval pipeline consists of multiple stages to improve retrieval quality.

### 1. Dense Retrieval

- Converts query into dense embeddings
- Retrieves similar chunks from FAISS

### 2. BM25 Retrieval

- Performs keyword-based retrieval
- Complements dense retrieval

### 3. Hybrid Retrieval

- Combines Dense + BM25
- Uses Reciprocal Rank Fusion (RRF)

### 4. Cross-Encoder Reranking

- Scores retrieved chunks jointly with the query
- Produces the final Top-K contexts

### 5. Response Generation

- Sends retrieved contexts to the local LLM
- Generates a grounded response with citations

---

# Evaluation

The project includes a dedicated evaluation framework.

## Retrieval Metrics

- Recall@K
- Mean Reciprocal Rank (MRR)

## Evaluation Dataset

- 28 manually curated questions
- Ground-truth answers
- Category labels
- Difficulty levels

Categories include:

- Definition
- Mechanism
- Motivation
- Experiment
- Comparison

---

# Dashboard

The Streamlit dashboard provides three major modules.

## Ask

- Ask questions
- View generated answers
- Inspect retrieved sources

## Papers

- Browse indexed papers
- View corpus statistics

## Evaluation

- Browse evaluation results
- Compare generated answers with ground truth
- Inspect retrieved contexts

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/research-paper-rag.git

cd research-paper-rag
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com

Pull the model:

```bash
ollama pull qwen3.5:4b
```

---

# Running the Project

## Step 1 — Index Papers

```bash
python -m src.ingest
```

---

## Step 2 — Launch the Dashboard

```bash
streamlit run app.py
```

---

## Step 3 — Run Evaluation

```bash
python -m evaluation.run_evaluation
```

---

# Configuration

Key configuration values are defined in `config.py`.

```python
Embedding Model : BAAI/bge-small-en-v1.5

Retriever : Hybrid (Dense + BM25)

Reranker : cross-encoder/ms-marco-MiniLM-L-6-v2

LLM : qwen3.5:4b

Top-K : 5
```

---

# Current Capabilities

- Multi-document question answering
- Citation-grounded responses
- Hybrid retrieval
- Local inference
- Retrieval evaluation
- Interactive dashboard

---

# Future Improvements

- Metadata filtering
- Query rewriting
- HyDE retrieval
- Context compression
- Streaming responses
- Chat history
- Docker deployment
- Cloud vector databases
- Automated answer evaluation with RAGAS

---

# Key Learnings

This project provided hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Dense vector embeddings
- Hybrid retrieval strategies
- Cross-Encoder reranking
- Prompt engineering
- Local LLM deployment using Ollama
- Evaluation of retrieval systems
- Building production-style AI applications

---

# Acknowledgements

This project builds upon ideas from:

- Attention Is All You Need
- BERT
- Dense Passage Retrieval (DPR)
- Retrieval-Augmented Generation (RAG)
- LoRA

---

# License

This project is released under the MIT License.

---

## Author

**Jasleen Kaur**

Final Year B.Tech (Computer Science Engineering)

Interested in Machine Learning, NLP, Retrieval-Augmented Generation, and Applied AI Systems.