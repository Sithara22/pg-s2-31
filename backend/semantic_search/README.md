# SiteSeer 2B — Semantic Search

This folder contains the semantic search prototype for SiteSeer Project 2B.

It converts learner questions and SiteSeer Knowledge Objects (KOs) into embeddings and uses vector similarity to retrieve the most relevant KOs.

## Current Technology

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding size:** 384 dimensions
- **Vector database:** PostgreSQL + pgvector
- **Similarity method:** Cosine similarity
- **Current data:** Synthetic development EKOs

MiniLM was selected as the current development model after comparison with `all-mpnet-base-v2`. Both achieved the same retrieval results on the current synthetic test set, while MiniLM was smaller and faster.

The model will be re-evaluated when real published SiteSeer EKOs are available.

## How It Works

```text
Search-ready EKOs
       ↓
MiniLM embeddings
       ↓
PostgreSQL + pgvector
       ↓
Learner question
       ↓
Question embedding
       ↓
Cosine similarity search
       ↓
Top-K Knowledge Objects
```

## Setup

Activate the project environment:

```bash
conda activate siteseer-2b
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Create a `.env` file in the project root using `.env.example`:

```text
DB_NAME=siteseer_2b
DB_USER=your_postgres_username
DB_HOST=localhost
DB_PORT=5432
```

Make sure PostgreSQL is running and the `siteseer_2b` database has the pgvector extension enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Run Semantic Search

### 1. Generate embeddings

```bash
python backend/semantic_search/generate_embeddings.py
```

### 2. Load the embeddings into pgvector

```bash
python backend/semantic_search/index_pgvector.py
```

### 3. Run semantic search

```bash
python backend/semantic_search/search_pgvector.py
```

Example question:

```text
How do I stop a newly installed wall frame from moving?
```

The system returns the Top-K Knowledge Objects that are semantically most relevant to the question.

## Evaluation

Run the retrieval test:

```bash
python backend/semantic_search/evaluate_retrieval.py
```

Compare embedding models:

```bash
python backend/semantic_search/compare_embedding_models.py
```

The current evaluation uses synthetic data, so the results are for development purposes only.

## Important

The current dataset contains **synthetic development EKOs** and is not authoritative construction advice.

In the final system, Project 2B should index only knowledge approved for AI use by Project 2A. The pgvector database is a derived search index; Project 2A remains the authoritative knowledge source.

## Status

- Data preprocessing: ✅
- Embedding generation: ✅
- PostgreSQL + pgvector: ✅
- Basic semantic search: ✅
- Embedding model evaluation: ✅
- Metadata filtering: ⬜ Next
- RAG / AI Tutor: ⬜ Later