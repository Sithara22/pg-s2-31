# Team 2A integration preparation

This adapter is provisional. Team 2A has not provided its APIs or schema. The
paths, authentication and payloads below are assumptions for local development,
not an agreed contract. No new public ingestion endpoint is exposed.

## Bulk ingestion

The assumed bulk endpoint accepts GET and returns a complete JSON snapshot:

```json
{"complete": true, "ekos": []}
```

Each EKO uses the existing mock fields (title, l1–l5, stage, trade,
jurisdiction, learner_level, publication_status, validation_status,
ai_indexable, superseded, version), plus logical_ko_id, is_current and
viewer_exposable. `ko_id` is provisionally the immutable EKO **version ID**;
the query-time endpoint must return IDs in that same namespace. IDs may contain
letters, numbers, underscores, periods, colons and hyphens.

Only published, validated, current, non-superseded, AI-indexable and
viewer-exposable records enter the index. Malformed records, duplicate version
IDs and multiple eligible versions of one logical object reject the snapshot.
Empty eligible snapshots are rejected to prevent accidental index erasure.
Pagination is deliberately unsupported: 2A must explicitly confirm completeness.

Validate the included synthetic fixture without PostgreSQL or Ollama:

```bash
python -m backend.integration.ingest --fixture data/mock/team2a_bulk.json
```

Add `--apply` to embed records and replace the entire derived pgvector index,
including removal of stale versions. This uses the existing database settings,
MiniLM and a single database transaction; validation and embedding occur before
database mutation. Do not point this at an index you need to preserve.
Omit `--fixture` to fetch TEAM2A_BULK_URL. Refresh is manually invoked for this
prototype; it is not a background job. A failed refresh preserves the old index.

## Query-time integration

`TEAM2A_MODE` in the server environment is the only filtering toggle:

- `disabled` (default): accept required 1A context and search the local index without 2A.
- `api`: send required 1A context to 2A and restrict retrieval using its eligible IDs.

Restart the backend after changing the environment. Requests cannot override this
setting; the former `use_2a_filtering` request field is rejected. There is no UI
toggle. Both modes require the same question and complete context object.
If api mode lacks TEAM2A_ELIGIBILITY_URL, requests return an explanatory 503;
the server can still start so the environment setting can be corrected.
Both URLs are full configurable endpoint URLs. An optional
TEAM2A_API_TOKEN supplies a Bearer header; requests have a 30-second timeout.
Use HTTPS for remote authenticated services. These settings load from root .env.

POST /ask requires `context` containing module_id, stage_id,
siteseer_node_id, selected_node_ids, jurisdiction, trade_or_domain and
learner_level. After resolving conversational references, the adapter POSTs
those fields and `question` to 2A. The assumed response is:

```json
{"eko_ids": ["EKO-001-v1"]}
```

The pgvector query intersects those IDs with any existing metadata filters
**before** ranking and limiting. Empty eligibility returns insufficient evidence;
unknown IDs cannot retrieve anything. Upstream failures or invalid responses
produce HTTP 503, with no unrestricted retrieval fallback. Extra context fields
are rejected until an agreed mapping exists. The demo frontend accepts pasted 1A context; automatic wiring from Team 1A remains a separate integration step.

### Team 1A request with filtering off for now

```json
{
  "question": "What evidence should I record?",
  "context": {
    "module_id": "module-1",
    "stage_id": "stage-1",
    "siteseer_node_id": "node-1",
    "selected_node_ids": ["node-1"],
    "jurisdiction": "SA",
    "trade_or_domain": "Carpentry",
    "learner_level": "Foundation"
  }
}
```

Send this JSON to POST /ask with TEAM2A_MODE=disabled for now. All seven
context fields shown above are required in both modes; selected_node_ids may be
empty, while string fields and node IDs must be non-empty. The exact 1A/2A
schema still needs confirmation. Off means semantic search uses the current local
index without applying 2A node/relationship restrictions. Existing top-level
metadata filters still work. Context is validated but isn't used to infer graph
relationships locally. Set TEAM2A_MODE=api once 2A is configured.
The local demo requires pasted 1A context JSON rather than inventing model data.

Run API tests with `python -m unittest backend.integration.test_filtering_api`
after installing FastAPI and httpx (httpx is needed only for these tests).

## Confirm with Team 2A

- Actual URLs, HTTP methods, authentication and error formats.
- Bulk response envelope, completeness/pagination and eligibility flags.
- EKO content and metadata mapping; version and logical ID semantics.
- Eligibility request fields, relationship constraints and returned ID namespace.
- Required context, empty-result semantics and refresh expectations.

Update `team2a.py` at this boundary when the contract arrives. No actual client
connection, real-data correctness or real service performance has been verified.

Run offline tests with `python -m unittest backend.integration.test_team2a`.
