# RAG Pipeline — PDF Question Answering with Evaluation

A full end-to-end Retrieval-Augmented Generation (RAG) project built over PDF documents. Covers every stage: ingestion → querying → test set generation → evaluation. Runs completely free using local models via Ollama.

---

## What We Built

### Stage 1 — PDF Ingestion (`ingest.py`)

Loads a PDF, splits it into chunks, embeds them, and stores them in a Pinecone vector index.

- Loads PDF page by page using `PyPDFLoader`
- Splits into 200-token chunks with 20-token overlap using `RecursiveCharacterTextSplitter`
- Embeds chunks using `BAAI/bge-small-en` (HuggingFace, runs on CPU)
- Creates a Pinecone serverless index (`cosine` similarity, 384 dimensions)
- Uploads chunks in batches of 100

**Source document**: `document/gk-book.pdf` (a General Knowledge book)

---

### Stage 2 — Interactive QA (`query.py`)

An interactive terminal chatbot that answers questions grounded in the indexed PDF.

- Takes a user question → retrieves top-5 most similar chunks from Pinecone
- Builds a context-grounded prompt (capped at 4000 chars to keep inference fast)
- Sends prompt to `llama3.2:3b` running locally via Ollama
- Returns a concise answer along with the source page numbers
- Strict prompt: model is told to answer *only* from the context, never from prior knowledge

---

### Stage 3 — Test Set Generation (`generate_test.py`)

Automatically creates a labeled QA validation set from the PDF — no manual annotation.

- Loads and chunks the same PDF (400-token chunks)
- Filters out short or digit-only chunks
- Randomly samples 30 good chunks
- For each chunk, calls **Groq's `llama-3.3-70b-versatile`** to generate one factual Q&A pair
- Rate-limited at 2.5s between calls to stay within Groq's free tier (30 req/min)
- Skips chunks where no useful fact exists
- Saves output to `validation_set.csv`: `question`, `ground_truth`, `page`, `source_text`

---

### Stage 4 — Evaluation (`eval.py`)

Runs the full RAG pipeline against the validation set and scores it on 4 standard RAG metrics — entirely locally using Ollama (no API costs).

**Step 1 — Run RAG on all questions**: feeds each question through the same retrieve → prompt → LLM pipeline as `query.py`, collecting answers and retrieved chunks.

**Step 2 — Score each row**: prompts `llama3.2:3b` itself to act as a judge and return JSON scores for 4 metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer supported by retrieved context? (vs. hallucinated) |
| **Answer Relevancy** | Does the answer address the question asked? |
| **Context Precision** | Were the retrieved chunks actually relevant to the question? |
| **Context Recall** | Did the context contain what was needed to answer correctly? |

Saves full results (per-question breakdown + aggregate scores) to `eval_results.json` and prints a visual bar chart with improvement tips.

**Last run (26 questions, `llama3.2:3b`):**
```
faithfulness:      0.23  BAD  → model answers from prior knowledge, not context
answer_relevancy:  0.73  OK
context_precision: 0.42  LOW  → irrelevant chunks retrieved
context_recall:    0.65  OK
```

---

## Project Structure

```
.
├── ingest.py              # Stage 1: PDF ingestion into Pinecone
├── query.py             # Stage 2: Interactive RAG chatbot
├── generate_ingest.py     # Stage 3: Auto-generate QA validation set
├── eval.py              # Stage 4: Evaluate RAG pipeline
├── test.ipynb           # Early exploration notebook
├── document/
│   └── gk-book.pdf      # Primary source document
├── validation_set.csv   # Generated QA pairs (ground truth)
├── eval_results.json    # Latest evaluation output
└── requirements.txt
```

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF loading | `langchain-community` PyPDFLoader |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | `BAAI/bge-small-en` via HuggingFace |
| Vector store | Pinecone (serverless, free tier) |
| Local LLM | `llama3.2:3b` via Ollama |
| Test generation LLM | Groq `llama-3.3-70b-versatile` (free tier) |

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed
- Pinecone account (free tier)
- Groq account (free tier, only needed for test generation)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file:
```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

### Start Ollama

```bash
ollama pull llama3.2:3b
ollama serve        # keep running in a separate terminal
```

---

## Running the Project

```bash
# Stage 1: Ingest PDF into Pinecone (run once)
python ingest.py

# Stage 2: Ask questions interactively
python query.py

# Stage 3: Generate evaluation dataset (run once)
python generate_test.py

# Stage 4: Evaluate the RAG pipeline
python eval.py
```

---

## What to Improve Next

- **Faithfulness is low (0.23)**: `llama3.2:3b` tends to answer from memorized knowledge. Try a stricter system prompt or switch to a larger model.
- **Context precision is low (0.42)**: Chunks retrieved aren't always relevant. Try smaller chunk sizes or add a reranker (e.g. cross-encoder).
- **Hybrid search**: Combine dense (vector) + sparse (BM25) retrieval to improve context recall.
