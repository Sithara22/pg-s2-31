# SiteSeer 2B : AI Query Layer - Knowledge Corpus Intelligence

## Project Overview

SiteSeer Project 2B develops an AI-assisted query layer that enables learners to ask construction-related questions and receive answers grounded in SiteSeer knowledge.

The prototype uses semantic search and Retrieval-Augmented Generation (RAG) to retrieve relevant Educational Knowledge Objects (EKOs), generate responses, and provide traceable knowledge-object citations.

**Current scope:** A working React/FastAPI prototype using synthetic EKO data, with API-based embeddings and answer generation. Integration with approved client EKOs remains pending.

## System Architecture

```text
Learner question (React frontend → FastAPI POST /ask)
      ↓
MiniLM question embedding via Hugging Face Inference API
      ↓
PostgreSQL + pgvector retrieval
      ↓
Top-K search-ready EKOs
      ↓
Qwen answer generation via Hugging Face or Groq API
      ↓
Structured answer or insufficient-evidence response
      ↓
Citation and response validation
      ↓
Insufficient evidence → PostgreSQL unanswered-question storage
      ↓
Frontend /unanswered_questions page (topic filters and pagination)
```

Only responses with status `insufficient_evidence` are saved as unanswered questions. Answered requests and validation failures are not added to this list.

## Technology Stack

| Component | Technology |
| --- | --- |
| Languages | Python and JavaScript |
| Backend | FastAPI |
| Frontend | React 19 + Vite |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), via Hugging Face `hf-inference` |
| Vector database | PostgreSQL + pgvector |
| Retrieval | Cosine similarity, optional metadata filters |
| Default generation provider | Hugging Face router: `Qwen/Qwen3-4B:featherless-ai` |
| Optional generation provider | Groq: `qwen/qwen3.8-27b` |
| Deployment | Local processes or Docker Compose for the API and database |

## Inference Configuration

Copy `.env.example` to `.env` in the repository root and configure the database settings and inference credentials:

```dotenv
HF_TOKEN=your_huggingface_token_here
LLM_PROVIDER=huggingface
GROQ_API_KEY=
```

Set `LLM_PROVIDER=groq` and supply `GROQ_API_KEY` to use Groq for answer generation. `HF_TOKEN` is required with either provider because MiniLM embeddings always use the Hugging Face Inference API. The application does not load MiniLM weights locally or require Ollama for its current question-answering pipeline. Inference requires network access and provider access to the configured models.

## Unanswered Questions

Open [Unanswered questions](http://127.0.0.1:5173/unanswered_questions) with the frontend and backend running. The page shows stored questions, topic counts, topic filters, pagination (10 questions per page), and a refresh action.

When `POST /ask` returns `insufficient_evidence`, the backend saves the question, extracted topics and keywords, request filters, reason, and creation timestamp in PostgreSQL. Topic classification uses a fixed keyword vocabulary, with `unclassified` as the fallback.

| Stored reason | Meaning |
| --- | --- |
| `no_matching_evidence` | Retrieval returned no matching EKOs, including when metadata filters exclude all results. |
| `insufficient_support` | EKOs were retrieved, but the model reported that the evidence did not adequately support an answer. |

### Database Migration

Apply `backend/migrations/001_unanswered_questions.sql` to the configured database before using unanswered-question storage or listing. For a local database, run from the repository root with your configured connection values:

```bash
psql -h localhost -p 5432 -U postgres -d siteseer_2b -f backend/migrations/001_unanswered_questions.sql
```

For Docker Compose, run the migration against the database service:

```bash
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < backend/migrations/001_unanswered_questions.sql
```

The Compose example uses Bash input redirection. The migration creates the table and indexes and is safe to reapply. API startup and Compose do not apply it automatically. If saving an insufficient-evidence request fails, `/ask` returns HTTP 503.

### Backend Endpoints

The API defaults to `http://127.0.0.1:8000`. Interactive API documentation is available at [FastAPI docs](http://127.0.0.1:8000/docs); it includes the following endpoints automatically.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | API liveness; does not verify the database or inference providers. |
| POST | `/ask` | Answer a question with optional `stage`, `trade`, `jurisdiction`, and `learner_level` filters; persist insufficient-evidence requests. |
| GET | `/unanswered-questions` | List stored questions, newest first, with optional topic filtering and pagination. |
| GET | `/unanswered-questions/topics` | List distinct topics and their question counts, ordered by count descending, then topic ascending. |

`GET /unanswered-questions` accepts `limit` (default 50, range 1–200), `offset` (default 0, minimum 0), and optional `topic` (1–100 characters). Topics are normalized to lowercase with whitespace collapsed, then matched as an exact topic label. A blank topic is rejected.

Example: `GET /unanswered-questions?limit=10&offset=0&topic=wall%20framing`.

The response contains `items`, `total`, `limit`, and `offset`. Each item contains `id`, `question`, `topics`, `keywords`, `filters`, `reason`, and `created_at`. The topics endpoint returns `items` containing `topic` and `question_count`. Listing endpoints return HTTP 503 on database failures; invalid query parameters return HTTP 422.

The frontend route uses an underscore (`/unanswered_questions`); backend routes use hyphens (`/unanswered-questions`).

## Documentation

- [Semantic Search](backend/semantic_search/README.md) — environment setup, embedding generation, pgvector indexing, semantic retrieval, and retrieval evaluation.
- [RAG Pipeline](backend/llm/README.md) — answer validation and historical local-model comparisons and evaluation.
- [Backend](backend/README.md) — FastAPI setup and Docker configuration.
- [Frontend](frontend/README.md) — React application setup.

Start with the semantic search guide to configure the database and index, then apply the unanswered-question migration above and start the backend and frontend. Some component guides still describe the earlier local Ollama setup; the inference configuration above reflects the current application. Run embedding and retrieval scripts as modules from the repository root, for example `python -m backend.semantic_search.generate_embeddings`.

## Development Status

The following components have been implemented using synthetic development data:

- Search-ready EKO preprocessing and embedding generation
- PostgreSQL + pgvector indexing and semantic retrieval
- Optional stage, trade, jurisdiction, and learner-level filters
- Hugging Face/Groq API-based Qwen answer generation
- Remote MiniLM sentence-transformer embeddings via Hugging Face
- End-to-end RAG answer generation
- Structured response and EKO citation validation
- Insufficient-evidence response handling
- Persistent unanswered-question storage with topic and keyword classification
- Backend listing endpoints and a frontend page for exploring knowledge gaps
- React/FastAPI integration and Docker Compose support for the API and database
- Synthetic benchmark evaluation

### Historical end-to-end synthetic benchmark (local Ollama)

The following committed results were produced on 12 October 2026 using local Ollama `qwen3:4b-q4_K_M`, before the migration to hosted inference. They do not measure the current Hugging Face/Groq configuration; provider-specific evaluation remains necessary.

| Metric | Result |
| --- | ---: |
| Total questions | 40 |
| Answerable questions answered | 35/35 |
| Insufficient-evidence questions refused | 5/5 |
| Retrieval Hit@1 (answerable cases) | 35/35 |
| Retrieval Hit@3 (answerable cases) | 35/35 |
| Citation validation failures | 0 |
| Invalid model responses | 0 |
| Mean end-to-end latency | 5.783 seconds |

Results are recorded in [`data/evaluation/results/end_to_end_qwen3_4b_prompt_v2.json`](data/evaluation/results/end_to_end_qwen3_4b_prompt_v2.json). Other evaluation artifacts are available in `data/evaluation/results/`.

**Limitations:** These results reflect automated checks on a synthetic corpus. Citation validation verifies that cited IDs are among retrieved EKOs; it does not independently prove that every generated claim is supported. Results are not production accuracy estimates.

## Next Steps

Integrate eligible, approved SiteSeer EKOs from Project 2A once the data contract is confirmed. Rebuild the derived index and re-evaluate retrieval, answer relevance, factual grounding, and refusal behavior using authorised client data.

## Development Notes

The PostgreSQL database and pgvector index are local, derived development resources; they are not the authoritative SiteSeer knowledge source. Consult the component READMEs for setup and reproducibility instructions.
