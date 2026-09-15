# SiteSeer 2B — Semantic Search

This folder contains the semantic search and retrieval prototype for SiteSeer Project 2B.

It converts learner questions and SiteSeer Knowledge Objects (KOs) into embeddings and retrieves the most relevant KOs using vector similarity and optional metadata filters.

## Current Technology

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding size:** 384 dimensions
- **Vector database:** PostgreSQL + pgvector
- **Similarity:** Cosine similarity
- **Current data:** Synthetic development EKOs

MiniLM was selected as the development model after comparison with `all-mpnet-base-v2`. The model will be re-evaluated when real published SiteSeer EKOs are available.

## How It Works

```text
Search-ready EKOs
       ↓
MiniLM embeddings
       ↓
PostgreSQL + pgvector
       ↓
Learner question + optional metadata
       ↓
Filter eligible KOs
       ↓
Cosine similarity search
       ↓
Top-K Knowledge Objects
```

Supported metadata filters:

- Stage
- Trade
- Jurisdiction
- Learner level

Metadata filtering restricts which KOs are eligible before semantic similarity ranking is performed.

## Setup

Activate the environment:

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

Make sure PostgreSQL is running and pgvector is enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Run Semantic Search

### 1. Generate embeddings

```bash
python backend/semantic_search/generate_embeddings.py
```

### 2. Load embeddings into pgvector

```bash
python backend/semantic_search/index_pgvector.py
```

### 3. Run semantic search

```bash
python backend/semantic_search/search_pgvector.py
```

Enter a learner question and optionally provide metadata filters.

Example:

```text
Question: How do I stop a newly installed wall frame from moving?

Stage: Wall Framing
Trade: Carpentry
Jurisdiction: SA
Learner level: Foundation
```

The system filters eligible KOs and then ranks them by semantic similarity.

## Evaluation

Run retrieval evaluation:

```bash
python backend/semantic_search/evaluate_retrieval.py
```

The current development test set contains:

- 10 answerable questions
- 6 metadata-filtered retrieval tests
- 5 insufficient-corpus questions reserved for later refusal testing

Current synthetic evaluation result:

```text
Hit@1: 10/10 (100%)
Hit@3: 10/10 (100%)
```

These results are based on a small synthetic dataset and do **not** represent final SiteSeer system accuracy.

To compare embedding models:

```bash
python backend/semantic_search/compare_embedding_models.py
```

## Important

The current dataset contains **synthetic development EKOs** and is not authoritative construction advice.

In the final system, Project 2B should index only knowledge approved for AI use by Project 2A.

Project 2A remains the authoritative knowledge source. The Project 2B pgvector database is a derived and rebuildable search index.

## Current Status

- Data preprocessing: ✅
- Embedding generation: ✅
- PostgreSQL + pgvector: ✅
- Semantic search: ✅
- Embedding model evaluation: ✅
- Metadata filtering: ✅
- Retrieval test dataset: ✅
- Answerability / refusal: ⬜ Next
- RAG / AI Tutor: ⬜ Later