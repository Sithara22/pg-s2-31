# Local demo backend

The FastAPI backend wraps the existing RAG pipeline. Start PostgreSQL and
Ollama with qwen3:4b-q4_K_M available, install backend/requirements.txt, then run
from the repository root:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Use one worker. MiniLM loads once at startup; each question opens and closes its
own database connection. GET /health checks API liveness only.

## Team 1A questions

POST /ask requires a question and a complete Team 1A context in both modes.
See the [request example and provisional contract](integration/README.md#team-1a-request-with-filtering-off-for-now).
Question-only requests now return HTTP 422. All seven context fields in the
example are required. This intentionally replaces the earlier optional context.

The only toggle is TEAM2A_MODE in the root .env or server environment:

- disabled: skip Team 2A and retrieve from the existing local index.
- api: send question/context to Team 2A and retrieve only eligible IDs.

Restart the backend after changing this setting. Callers cannot override it.
The use_2a_filtering request field is no longer accepted. Context remains
required and validated even when Team 2A is disabled. Existing optional top-level
stage, trade, jurisdiction and learner_level filters still narrow local search.

The response remains answer, status, sources and retrieval. Status is answered,
insufficient_evidence, invalid_model_response or citation_validation_failed.
Invalid requests return 422; unavailable dependencies return 503. An empty 2A
allowlist returns insufficient evidence without unrestricted search.

## Conversation context

An optional history array accepts up to six messages (three exchanges), each
with role user/assistant and content of at most 6000 characters. The current
question is excluded from history. Qwen resolves follow-up references before
fresh retrieval. Prior answers are not evidence. The server stores no chats.

## Browser access

CORS permits localhost and 127.0.0.1 on ports 5173 and 3000. The API runs on
port 8000 by default; /docs exposes the request schema. Any different Team 1A
origin must be explicitly configured before browser integration.
