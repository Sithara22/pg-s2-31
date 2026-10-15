# Local demo backend

This minimal FastAPI application wraps the existing RAG pipeline. It uses the
same `.env`, PostgreSQL index and Ollama model as the working CLI. Run commands
from the repository root, in your existing RAG Python environment (Python 3.10+).

## Install and start

Keep PostgreSQL and Ollama running with `qwen3:4b-q4_K_M` available, then run:

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`. MiniLM loads once at startup and is
reused for questions. Each question opens and closes its own database connection.
Stop with Ctrl+C. Restart after code changes. Use one server worker for this demo.

Open http://127.0.0.1:8000/docs to try requests in the browser:
expand `POST /ask`, click **Try it out**, enter a request and click **Execute**.

## Endpoints

- `GET /health`: returns `{"status":"ok"}` when the API is running. This is not
  a PostgreSQL/Ollama readiness check.
- `POST /ask`: accepts a question and optional exact-match metadata filters.
  Retrieval uses the pipeline's default top three results.

Minimal request:

```json
{"question": "How do I stop a newly installed wall frame from moving?"}
```

Request with optional filters:

```json
{
  "question": "How do I stop a newly installed wall frame from moving?",
  "stage": "Wall Framing",
  "trade": "Carpentry",
  "jurisdiction": "SA",
  "learner_level": "Foundation"
}
```

Terminal example:

```bash
curl http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I stop a newly installed wall frame from moving?"}'
```

Response fields:

| Field | Meaning |
|---|---|
| `answer` | Generated answer or the pipeline's fallback/refusal text |
| `status` | `answered`, `insufficient_evidence`, `invalid_model_response`, or `citation_validation_failed` |
| `sources` | List of cited EKO IDs |
| `retrieval` | Retrieved EKO IDs, titles and similarity scores |

The frontend can match `sources` against `retrieval` to show cited source titles.
Full EKO text is not returned by this minimal endpoint. Refusals and validation
fallbacks return HTTP 200 with the corresponding `status`. Invalid input returns
HTTP 422. Pipeline exceptions return HTTP 503 with a `detail` message; the full
error is logged in the server terminal.

## Frontend integration later

The API allows browser requests from `localhost` and `127.0.0.1` on ports 5173
and 3000. Use `http://127.0.0.1:8000` as the frontend API base URL. Send JSON to
`/ask`, show a loading state while waiting, then display the answer and sources.
Check HTTP errors before reading an answer. This is a single-question local demo
using synthetic knowledge, without conversation memory or authentication.
