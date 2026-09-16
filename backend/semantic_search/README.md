# SiteSeer 2B — Semantic Search

This folder contains the semantic search and retrieval prototype for SiteSeer Project 2B.

It converts learner questions and SiteSeer Knowledge Objects (KOs) into embeddings and retrieves the most relevant KOs using vector similarity and optional metadata filters.

## Current Technology

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding size:** 384 dimensions
- **Vector database:** PostgreSQL + pgvector
- **Similarity:** Cosine similarity
- **Current data:** Synthetic development EKOs

MiniLM was selected as the development embedding model after comparison with `all-mpnet-base-v2`.

Both models achieved the same retrieval performance on the current synthetic evaluation set. MiniLM was selected because it uses smaller 384-dimensional embeddings and was lighter and faster during development.

The embedding model will be re-evaluated when real published SiteSeer EKOs are available.

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

---

## Setup

### 1. Clone the Repository

Clone the project repository and move into the project folder.

```bash
git clone https://github.com/Sithara22/pg-s2-31.git
cd pg-s2-31
```

---

### 2. Activate the Python Environment

The current development environment uses Conda with Python 3.11.

If the environment already exists:

```bash
conda activate siteseer-2b
```

If setting up the environment for the first time:

```bash
conda create -n siteseer-2b python=3.11 -y
conda activate siteseer-2b
```

Install the project dependencies:

```bash
pip install -r backend/requirements.txt
```

---

### 3. Set Up PostgreSQL

The PostgreSQL database is stored **locally on each developer's computer**.

The database itself is **not stored in GitHub**.

Each developer must create their own local SiteSeer 2B PostgreSQL database.

Make sure PostgreSQL is installed and running before continuing.

Open PostgreSQL:

```bash
psql postgres
```

Check your PostgreSQL username:

```sql
SELECT current_user;
```

You can also view the available PostgreSQL users with:

```text
\du
```

Create the SiteSeer 2B database:

```sql
CREATE DATABASE siteseer_2b;
```

Connect to the database:

```text
\c siteseer_2b
```

Enable the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Exit PostgreSQL:

```text
\q
```

---

### 4. Configure the Database Connection

Create a `.env` file in the project root based on `.env.example`.

Example:

```text
DB_NAME=siteseer_2b
DB_USER=your_postgres_username
DB_HOST=localhost
DB_PORT=5432
```

Replace:

```text
your_postgres_username
```

with the PostgreSQL username returned by:

```sql
SELECT current_user;
```

---

### 5. Generate Embeddings

Generate embeddings for the current search-ready EKO dataset:

```bash
python backend/semantic_search/generate_embeddings.py
```

The current development embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The generated embeddings are local derived artifacts and are not required to be stored in GitHub.

---

### 6. Build the Local pgvector Index

Load the generated embeddings into PostgreSQL:

```bash
python backend/semantic_search/index_pgvector.py
```

The indexing script creates and populates the local:

```text
knowledge_embeddings
```

table.

The table contains the Knowledge Object identifiers, searchable content, metadata and vector embeddings required for semantic retrieval.

The pgvector table is a **derived and rebuildable search index**. It is not the authoritative SiteSeer knowledge database.

---

### 7. Verify the Database

Open the local SiteSeer 2B database:

```bash
psql siteseer_2b
```

List the tables:

```text
\dt
```

You should see a table similar to:

```text
Schema |         Name         | Type  | Owner
-------+----------------------+-------+----------------
public | knowledge_embeddings | table | <your_username>
```

The owner will normally be the PostgreSQL username of the developer who created the local database.

Check how many Knowledge Objects are currently indexed:

```sql
SELECT COUNT(*) FROM knowledge_embeddings;
```

You can also inspect the indexed records:

```sql
SELECT ko_id, title
FROM knowledge_embeddings
ORDER BY ko_id;
```

Exit PostgreSQL:

```text
\q
```

> **Important:** Each developer has their own local PostgreSQL database. Cloning or pulling the GitHub repository does not copy another developer's `siteseer_2b` database or `knowledge_embeddings` table. The local vector index can be rebuilt from the project data using the embedding and indexing scripts.

---

## Run Semantic Search

After completing the setup and building the local vector index, run:

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

The system first restricts the eligible Knowledge Objects using any supplied metadata filters and then ranks the remaining KOs using cosine similarity.

Leaving a metadata filter blank means that filter is not applied.

---

## Retrieval Evaluation

Run the semantic retrieval evaluation:

```bash
python backend/semantic_search/evaluate_retrieval.py
```

The current development test set contains:

- 10 answerable retrieval questions
- 6 metadata-filtered retrieval tests
- 5 insufficient-corpus questions reserved for answerability/refusal testing

Current synthetic retrieval result:

```text
Hit@1: 10/10 (100%)
Hit@3: 10/10 (100%)
```

These results are based on a small synthetic development dataset and do **not** represent final SiteSeer system accuracy.

The retrieval system must be re-evaluated when a larger development corpus and real published SiteSeer EKOs become available.

---

## Embedding Model Evaluation

To compare candidate embedding models:

```bash
python backend/semantic_search/compare_embedding_models.py
```

The current development comparison includes:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

Both models achieved the same retrieval ranking performance on the current synthetic test set.

MiniLM was selected provisionally because it produces smaller embeddings and demonstrated lower computational overhead during the current development evaluation.

This decision is not final and should be re-evaluated using real SiteSeer data.

---

## Important Development Notes

- Current construction content is synthetic development data.
- Current retrieval results are not production accuracy measurements.
- PostgreSQL databases are local to each developer and are not stored in GitHub.
- Generated embeddings are derived artifacts and can be regenerated.
- The pgvector index can be rebuilt from the approved search-ready dataset.
- Project 2A remains the authoritative knowledge source.
- Embedding model selection is provisional.
- Similarity scores are not probabilities or accuracy percentages.
- Current answerability scores must not be used as final production thresholds.
- Real SiteSeer EKOs will require re-indexing and re-evaluation.

---

## Current Status

- Data preprocessing: ✅
- Search-ready dataset generation: ✅
- Embedding generation: ✅
- PostgreSQL + pgvector: ✅
- Semantic search: ✅
- Embedding model evaluation: ✅
- Metadata filtering: ✅
- Retrieval test dataset: ✅
- Semantic retrieval evaluation: ✅
- Initial answerability analysis: ✅
- Expanded synthetic EKO corpus: ⬜ Next
- Answerability / refusal mechanism: ⬜
- RAG / AI Tutor: ⬜