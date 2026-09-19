# Research Paper RAG

A local **Retrieval-Augmented Generation (RAG)** system for question answering over Machine Learning and NLP research papers (and user-uploaded PDFs).

The pipeline combines **dense semantic search**, **BM25 lexical search**, **Reciprocal Rank Fusion (RRF)**, **cross-encoder reranking**, and a **local or hosted LLM** to produce citation-grounded answers with document titles and page numbers.

**Repository:** [github.com/jasleenjk07/research-paper-rag](https://github.com/jasleenjk07/research-paper-rag)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Module Reference](#module-reference)
- [Setup](#setup)
- [Usage](#usage)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Configuration](#configuration)
- [LLM Providers](#llm-providers)
- [Retrieval Details](#retrieval-details)
- [Answer Generation](#answer-generation)
- [Evaluation](#evaluation)
- [Data and Storage](#data-and-storage)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Author](#author)

---

## Features

- Ingest multiple research PDFs from `data/papers/` or the dashboard uploader
- Incremental ingestion (skips already-processed filenames)
- Recursive character chunking (`CHUNK_SIZE=750`, `CHUNK_OVERLAP=150`)
- Dense retrieval with **BAAI/bge-small-en-v1.5** + **FAISS** (`IndexFlatIP` + L2 normalization ≈ cosine similarity)
- Lexical retrieval with **BM25** (`rank-bm25`)
- Hybrid fusion with **Reciprocal Rank Fusion** (`RRF_K=60`)
- Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Optional **per-document** retrieval scope (search one PDF or the full corpus)
- Citation-grounded answers: `[Document Title, Page X]`
- Local LLM via **Ollama** (`qwen3.5:4b`) or cloud via **Groq**
- Interactive Streamlit UI: Ask / Papers / Evaluation
- Retrieval evaluation scripts: **Recall@K** and **MRR**
- 28 curated evaluation questions with ground-truth answers

---

## Architecture

```text
PDFs (data/papers/ or uploads/)
        │
        ▼
   PDF Loader (titles + pages)
        │
        ▼
 RecursiveCharacterTextSplitter
        │
        ▼
 BGE Small embeddings → FAISS index
        │                 + BM25 corpus
        │
Query ──┤
        ├─ Dense retrieve (top 20)
        ├─ BM25 retrieve (top 20)
        │
        ▼
 Reciprocal Rank Fusion (RRF_K=60)
        │
        ▼
 Cross-Encoder rerank → top 5 chunks
        │
        ▼
 LLM (Ollama / Groq) → grounded answer + citations
```

**Production query path** (`RAGPipeline.answer`):

1. `retrieve_hybrid(..., retrieve_k=20, top_k=20)` — dense + BM25, fused with RRF
2. `rerank_documents(..., top_k=5)` — cross-encoder scores query–passage pairs
3. `generate_answer(...)` — LLM answers only from retrieved context

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| UI | Streamlit |
| PDF loading | LangChain `PyPDFLoader` / `pypdf` |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Embeddings | `sentence-transformers` — `BAAI/bge-small-en-v1.5` |
| Vector store | `faiss-cpu` |
| Lexical retrieval | `rank-bm25` (`BM25Okapi`) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | `CrossEncoder` — `ms-marco-MiniLM-L-6-v2` |
| LLM (local) | Ollama + `langchain-ollama` (`qwen3.5:4b`) |
| LLM (cloud) | Groq via `langchain-openai` OpenAI-compatible API |
| Evaluation | Custom Recall@K / MRR; optional RAGAS (`evaluate_answers.py`) |

---

## Project Structure

```text
research-paper-rag/
├── app.py                      # Streamlit dashboard
├── main.py                     # Simple CLI demo
├── run_dashboard.sh            # Launch Streamlit with project venv
├── requirements.txt
├── README.md
│
├── data/
│   └── papers/                 # Built-in research PDFs
│
├── uploads/                    # User uploads from the dashboard (gitignored)
│
├── src/
│   ├── config.py               # All tunable constants
│   ├── loader.py               # PDF loading + title extraction
│   ├── chunker.py              # Text splitting + chunk_id metadata
│   ├── embeddings.py           # SentenceTransformer loader
│   ├── vector_store.py         # FAISS build / add / save / load
│   ├── ingest.py               # Incremental PDF ingestion
│   ├── retriever.py            # Dense FAISS retrieval
│   ├── bm25_retriever.py       # BM25 index + retrieval
│   ├── hybrid_retriever.py     # Dense + BM25 + RRF
│   ├── reranker.py             # Cross-encoder reranking
│   ├── generator.py            # Prompt construction + answer generation
│   ├── llm.py                  # Ollama / Groq provider factory
│   └── rag_pipeline.py         # End-to-end orchestration
│
├── evaluation/
│   ├── questions.json          # 28 curated Q&A items
│   ├── run_evaluation.py       # Full pipeline run → results.json
│   ├── evaluate_retrieval.py   # Dense-only Recall@K / MRR
│   ├── evaluate_bm25.py        # BM25-only metrics
│   ├── evaluate_hybrid.py      # Hybrid (RRF) metrics
│   ├── evaluate_reranker.py    # Hybrid + rerank metrics
│   ├── evaluate_answers.py     # RAGAS answer quality (extra deps)
│   └── debug_retrieval.py      # Interactive dense retrieval debug
│
├── .streamlit/
│   └── config.toml             # fileWatcherType=none, no usage stats
│
└── storage/                    # Created after ingestion (gitignored artifacts)
    ├── faiss.index
    ├── chunks.pkl
    └── processed_papers.json
```

---

## Module Reference

| Module | Responsibility |
|--------|----------------|
| `src/loader.py` | Load PDFs page-by-page; extract clean titles (research papers and USPTO-style patents); set `paper_name`, `page`, `title` metadata |
| `src/chunker.py` | Split documents; assign global `chunk_id` |
| `src/embeddings.py` | Load `SentenceTransformer(EMBEDDING_MODEL)` |
| `src/vector_store.py` | Encode chunks, L2-normalize, build/add FAISS `IndexFlatIP`, pickle chunks |
| `src/ingest.py` | Incremental ingest for `data/papers` and dashboard uploads |
| `src/retriever.py` | Dense query embedding + FAISS search; optional document filter |
| `src/bm25_retriever.py` | Whitespace-tokenized BM25Okapi corpus and scoring |
| `src/hybrid_retriever.py` | Run dense + BM25, fuse with RRF by `chunk_id` |
| `src/reranker.py` | CrossEncoder predict + sort + top-k |
| `src/generator.py` | Build context/prompt; invoke LLM; strip accidental `Answer:` prefix |
| `src/llm.py` | Resolve secrets/env; return ChatOllama or Groq ChatOpenAI |
| `src/rag_pipeline.py` | Load models/index once; `answer(question, document_name=None)` |
| `app.py` | Streamlit UI with upload, document scope, sources, and eval browser |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/jasleenjk07/research-paper-rag.git
cd research-paper-rag
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

> Always use this `venv` for `pip`, `python`, and `streamlit`. Running Streamlit from Conda base (or another interpreter) commonly causes `ModuleNotFoundError` for packages such as `sentence_transformers`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama (local LLM)

1. Install from [https://ollama.com](https://ollama.com)
2. Pull the default model:

```bash
ollama pull qwen3.5:4b
```

### 5. Index the built-in papers

```bash
python -m src.ingest
```

This reads PDFs from `data/papers/`, builds or updates the FAISS index, and writes:

- `storage/faiss.index`
- `storage/chunks.pkl`
- `storage/processed_papers.json`

---

## Usage

### Launch the dashboard

Preferred (forces the project venv):

```bash
./run_dashboard.sh
```

Or manually:

```bash
source venv/bin/activate
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

### CLI demo

```bash
python main.py
```

Runs one hardcoded question through `RAGPipeline` and prints the answer. Requires a built vector store.

### Re-ingest / add papers

```bash
# Add PDFs to data/papers/, then:
python -m src.ingest
```

Only **new filenames** (not already listed in `processed_papers.json`) are processed. Failed PDFs are skipped; the store is saved only if at least one file succeeds.

Uploads from the dashboard go to `uploads/` and use the same incremental ingest path (`ingest_uploaded_files`).

---

## Streamlit Dashboard

Sidebar navigation:

| Page | Description |
|------|-------------|
| **Ask** | Upload PDFs, choose document scope (`All documents` or one file), ask questions, view the answer and ranked sources (title, page, chunk ID, rerank score, preview) |
| **Papers** | Corpus size and list of processed documents; shows `FINAL_TOP_K` / `HYBRID_RETRIEVE_K` |
| **Evaluation** | Browse `evaluation/results.json` (after `run_evaluation.py`): filters by category/difficulty, generated answers, ground truth, citations, retrieved contexts |

Sidebar also shows the active embedding model, LLM provider, reranker, and k settings.

`.streamlit/config.toml` disables Streamlit’s file watcher (`fileWatcherType = "none"`) to avoid noisy reloads with PyTorch / Transformers.

---

## Configuration

All defaults live in `src/config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model |
| `CHUNK_SIZE` | `750` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `HYBRID_RETRIEVE_K` | `20` | Candidates from dense, BM25, and RRF before rerank |
| `FINAL_TOP_K` | `5` | Chunks kept after reranking (sent to the LLM) |
| `RRF_K` | `60` | RRF constant in `1 / (RRF_K + rank)` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `LLM_MODEL` | `qwen3.5:4b` | Ollama model name |
| `FAISS_INDEX_PATH` | `storage/faiss.index` | FAISS index path |
| `CHUNKS_PATH` | `storage/chunks.pkl` | Serialized chunks |
| `PROCESSED_PAPERS_PATH` | `storage/processed_papers.json` | Ingested filename list |

---

## LLM Providers

Configured in `src/llm.py`. Values are read from **Streamlit secrets** first, then **environment variables**.

### Local (default)

```bash
export LLM_PROVIDER=ollama
# Uses LLM_MODEL from src/config.py → qwen3.5:4b
```

Uses `ChatOllama` with `temperature=0` and `reasoning=False`.

### Groq (deployment)

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key
export GROQ_MODEL=qwen/qwen3.6-27b   # optional; this is the default
```

Or `.streamlit/secrets.toml` (gitignored):

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your_key"
GROQ_MODEL = "qwen/qwen3.6-27b"
```

Groq client settings: OpenAI-compatible base URL, `max_completion_tokens=700`, `reasoning_effort="none"`.

Unsupported `LLM_PROVIDER` values raise `ValueError` (supported: `ollama`, `groq`).

---

## Retrieval Details

### Dense retrieval

1. Encode the query with the embedding model
2. Convert to `float32` and L2-normalize
3. Search FAISS `IndexFlatIP`
4. Optional `document_name` filter: match `metadata["paper_name"]` to the filename stem (case-insensitive)

### BM25

1. Tokenize with lowercase whitespace split
2. Score with `BM25Okapi`
3. Same optional document filter as dense retrieval

### Reciprocal Rank Fusion

For each of dense and BM25 ranked lists:

```text
score[chunk_id] += 1 / (RRF_K + rank)     # RRF_K = 60, ranks start at 1
```

Chunks are merged by `chunk_id`, sorted by fused score, and truncated to `top_k`.

### Reranking

The cross-encoder scores `(question, passage)` pairs and returns the top `FINAL_TOP_K` documents with reranker scores.

---

## Answer Generation

`src/generator.py` builds a strict grounded prompt:

- Use **only** retrieved context; no outside knowledge
- Cite important claims as **`[Document Title, Page X]`** using titles/pages from context
- Place citations immediately after the claim
- If context is insufficient, reply exactly that the documents do not contain enough information
- Avoid openings like “Based on the provided context…” and avoid an `Answer:` heading

Context blocks are formatted as:

```text
[Title, Page N]

<chunk text>
```

Pages are stored 0-based in metadata and shown as `page + 1` in prompts and the UI.

---

## Evaluation

### Dataset (`evaluation/questions.json`)

**28** questions. Each item has:

| Field | Description |
|-------|-------------|
| `question` | Natural-language query |
| `expected_paper` | Filename stem that should be retrieved |
| `category` | `definition` / `mechanism` / `motivation` / `experiment` / `comparison` |
| `difficulty` | `easy` / `medium` / `hard` |
| `ground_truth` | Reference answer |

Approximate breakdown: definition (13), mechanism (5), motivation (4), experiment (3), comparison (3); difficulties easy (13), medium (9), hard (6).

Papers covered by the eval set include Attention Is All You Need, LoRA, DPR, RAG (NeurIPS 2020), and BERT. The corpus may also include Sentence-BERT and uploaded patents that are not in the eval set.

### Hit definition

A question is a hit if any of the top-k retrieved chunks has `metadata["paper_name"] == expected_paper`. Rank of the first such chunk is used for MRR.

### Scripts

| Command | What it measures |
|---------|------------------|
| `python -m evaluation.evaluate_retrieval` | Dense-only Recall@K + MRR |
| `python -m evaluation.evaluate_bm25` | BM25-only Recall@K + MRR |
| `python -m evaluation.evaluate_hybrid` | Hybrid RRF (no rerank) Recall@K + MRR |
| `python -m evaluation.evaluate_reranker` | Hybrid → rerank → top-5 Recall@K + MRR |
| `python -m evaluation.run_evaluation` | Full RAG answers → `evaluation/results.json` |
| `python -m evaluation.evaluate_answers` | RAGAS faithfulness / relevancy / context metrics → `evaluation/generated_answers.json` |
| `python -m evaluation.debug_retrieval` | Interactive dense retrieval inspection |

> `evaluate_answers.py` needs extra packages (`ragas`, `datasets`) that are **not** listed in `requirements.txt`. Install them separately if you run that script.

---

## Data and Storage

| Path | Role |
|------|------|
| `data/papers/*.pdf` | Built-in corpus |
| `uploads/` | Dashboard uploads (gitignored) |
| `storage/faiss.index` | FAISS index (gitignored) |
| `storage/chunks.pkl` | Pickled LangChain documents (gitignored) |
| `storage/processed_papers.json` | Sorted list of ingested **filenames** (gitignored) |

**Default corpus PDFs** (under `data/papers/`):

- `NIPS-2017-attention-is-all-you-need-Paper.pdf`
- `N19-1423- BERT.pdf`
- `report017-sentence-bert.pdf`
- `2020.emnlp-main.550-Dense-Passage-Retrieval.pdf`
- `NeurIPS-2020-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks-Paper.pdf`
- `2106.09685v2-LoRA.pdf`

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'sentence_transformers'`

Wrong Python environment. Activate `venv` or run:

```bash
./run_dashboard.sh
```

### `Vector store not found`

```bash
python -m src.ingest
```

### Streamlit / Transformers watcher noise or `torchvision` errors

Install full requirements (includes `torch` + `torchvision`) and keep `.streamlit/config.toml` with `fileWatcherType = "none"`. Prefer `./run_dashboard.sh`.

### Git push: `SSL certificate problem: self signed certificate in certificate chain`

Usually a VPN, proxy, or network HTTPS interceptor — not a repo bug. Try another network, disable VPN, or use SSH:

```bash
git remote set-url origin git@github.com:jasleenjk07/research-paper-rag.git
git push
```

### First-run model downloads

Embedding and reranker weights download from Hugging Face on first use. Unauthenticated requests may be rate-limited; set `HF_TOKEN` if needed.

### Ollama connection errors

Ensure the Ollama app/daemon is running and `qwen3.5:4b` is pulled before asking questions with `LLM_PROVIDER=ollama`.

---

## Future Improvements

- Query rewriting / HyDE
- Metadata filters beyond single-document scope
- Context compression and streaming responses
- Chat history in the dashboard
- Docker packaging
- Separate retrieval vs generation timing in `run_evaluation.py`
- Wire RAGAS into `requirements.txt` and CI
- Automated answer evaluation dashboards

---

## Acknowledgements

This project builds on ideas from:

- Attention Is All You Need
- BERT / Sentence-BERT
- Dense Passage Retrieval (DPR)
- Retrieval-Augmented Generation (RAG)
- LoRA (Low-Rank Adaptation)

---

## License

MIT

---

## Author

**Jasleen Kaur** — B.Tech Computer Science Engineering

Interests: Machine Learning, NLP, Retrieval-Augmented Generation, and applied AI systems.
