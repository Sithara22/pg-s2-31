# SiteSeer 2B — Local RAG Pipeline

This folder contains the **local retrieval-augmented generation (RAG) prototype** for SiteSeer Project 2B. It answers learner questions using retrieved Educational Knowledge Object (EKO) evidence and a locally hosted language model. The current implementation and evaluations use **synthetic development EKOs**, not approved production SiteSeer knowledge.

For embedding generation, PostgreSQL configuration, indexing, and retrieval setup, see [Semantic Search README](../semantic_search/README.md).

## Current Technology

| Component | Current implementation |
| --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Retrieval | PostgreSQL + pgvector, cosine similarity |
| Generation model | `qwen3:4b-q4_K_M` via local Ollama |
| Ollama endpoint | `http://127.0.0.1:11434/api/chat` |
| Generation options | `temperature=0`, `num_predict=512`, `think=false`, non-streaming |
| Retrieval depth | Top 3 by default; configurable |
| Response format | JSON object with `status` and `answer` |
| Current data | Synthetic, search-ready EKOs |

Qwen3 4B is the **provisional local generation model** based on development comparisons with Gemma3 4B and GPT-4.1 mini. Selection must be revisited with representative, approved SiteSeer data and requirements.

## How It Works

```text
Learner question + optional metadata filters
                   |
                   v
         MiniLM query embedding
                   |
                   v
     PostgreSQL + pgvector retrieval
                   |
                   v
      Top-K search-ready EKO evidence
                   |
                   v
       System prompt + EKO context
                   |
                   v
           Ollama / Qwen3 4B
                   |
                   v
         Structured JSON response
                   |
                   v
       Status and citation validation
                   |
                   v
  Answer + cited EKO IDs, or safe fallback
```

The retrieval layer supports optional **stage, trade, jurisdiction, and learner-level** filters. The generation step receives the question and retrieved EKO evidence; it is instructed not to treat evidence text as executable instructions, invent standards, or cite IDs outside the retrieved set.

## Prerequisites and Setup

Run commands from the **repository root**.

1. Complete the local Python environment, PostgreSQL/pgvector, embedding generation, and indexing instructions in the [Semantic Search README](../semantic_search/README.md). The local database is not included in GitHub and must be built on each developer's machine.
2. Install and start **Ollama** on your machine. Ensure its local service is accessible at `http://127.0.0.1:11434`.
3. Pull the selected generation model:

   ```bash
   ollama pull qwen3:4b-q4_K_M
   ```

4. Verify the model is installed:

   ```bash
   ollama list
   ```

The RAG pipeline uses the model name defined by `OLLAMA_MODEL` in `backend/llm/rag_pipeline.py`. Installing other Ollama models does **not** automatically change the model used by the pipeline.

**Local resources:** The selected model was run on an Apple Silicon development laptop. Runtime and memory requirements may differ across machines; the repository does not establish a universal minimum hardware specification. Local inference avoids per-request hosted-model API charges but still uses local compute resources.

## Run the End-to-End RAG Pipeline

After the local pgvector index and Ollama service are ready:

```bash
python -m backend.llm.rag_pipeline --question "What design inputs should be considered for residential HVAC capacity selection?"
```

The command prints the embedding model, generation model, retrieved EKO IDs and similarity scores, generated answer, cited sources, and final answer status.

To change the number of retrieved results:

```bash
python -m backend.llm.rag_pipeline --question "What design inputs should be considered for residential HVAC capacity selection?" --top-k 3
```

Optional filters are available through `--stage`, `--trade`, `--jurisdiction`, and `--learner-level`. Use values matching the indexed EKO metadata. Filters restrict the evidence available for answering and may result in an insufficient-evidence response.

## Answer Status, Citations, and Safety Checks

The model is instructed to return exactly one JSON object:

```json
{
  "status": "answered",
  "answer": "A response grounded in supplied evidence [MOCK-KO-037]."
}
```

The pipeline recognizes these internal statuses:

| Status | Meaning |
| --- | --- |
| `answered` | The model supplied an answer with at least one syntactically valid citation to a retrieved EKO. |
| `insufficient_evidence` | The available evidence does not adequately support an answer; no EKO citations are returned. |
| `invalid_model_response` | The model output failed structured-response validation; a safe fallback is returned. |
| `citation_validation_failed` | The answer failed citation validation; a safe fallback is returned. |

The citation validator checks that an answered response cites retrieved EKO IDs using the expected `[MOCK-KO-###]` format, and rejects unrecognized or malformed references. It does **not** prove that every factual claim is supported by the cited EKO. This distinction matters when interpreting evaluation results.

The system prompt also prioritizes evidence directly relevant to the learner's question rather than adding tangential material from other retrieved EKOs.

## Manual RAG Smoke Tests (Verified on Synthetic EKOs)

These checks were executed manually from the repository root using the local MiniLM + PostgreSQL/pgvector + Ollama/Qwen3 setup. They are intended as **quick, reproducible functional checks**, not a substitute for the 40-case automated evaluation or an independent factual-grounding audit. Ensure the prerequisites above are running before testing.

### Test 1 — End-to-end answer generation

**Purpose:** Check that a learner question flows through embedding, retrieval, generation, and citation validation.

```bash
python -m backend.llm.rag_pipeline --question "What evidence should be recorded before a concrete slab pour?"
```

**Observed result:** The system retrieved `MOCK-KO-001`, `MOCK-KO-005`, and `MOCK-KO-009`, produced an answer citing those three retrieved IDs, and returned `Answer status: answered`.

**Pass criteria:** Relevant evidence is retrieved; the pipeline returns `answered` with citations drawn from retrieved EKOs. **Limitation:** This run checks execution and citation-ID validity; it does not establish that every sentence fully answers the question or is supported by source text.

### Test 2 — Refusal when exact specifications are unsupported

**Purpose:** Check that the model does not invent an exact construction requirement absent from the retrieved evidence.

```bash
python -m backend.llm.rag_pipeline --question "What exact bolt diameter and spacing must be used for temporary wall bracing under the Australian Standards?"
```

**Observed result:** Three related EKOs were retrieved, but the answer stated that the evidence did not specify the exact dimensions. `Cited sources: None`; `Answer status: insufficient_evidence`.

**Pass criteria:** No fabricated bolt sizes, spacing, or standards requirements; an explicit insufficient-evidence response.

### Test 3 — Positive metadata filtering (stage + trade)

**Purpose:** Confirm that the search returns only EKOs matching both supplied metadata filters, before semantic ranking.

```bash
python -m backend.llm.rag_pipeline --question "Why is temporary wall bracing important?" --stage "Wall Framing" --trade "Carpentry"
```

**Observed RAG result:** `MOCK-KO-002`, `MOCK-KO-017`, and `MOCK-KO-016` were retrieved, and the answer cited `MOCK-KO-002` with status `answered`.

**Independent metadata check:** Run the interactive retrieval CLI:

```bash
python -m backend.semantic_search.search_pgvector
```

Enter the same question, `Wall Framing` for **Stage**, `Carpentry` for **Trade**, and leave the other filters blank. The verified run returned:

| Retrieved EKO | Stage | Trade | Similarity |
| --- | --- | --- | ---: |
| `MOCK-KO-002` | Wall Framing | Carpentry | 0.7027 |
| `MOCK-KO-017` | Wall Framing | Carpentry | 0.3667 |
| `MOCK-KO-016` | Wall Framing | Carpentry | 0.3603 |

**Pass criteria:** Every returned EKO matches **both** filters. Code inspection also confirmed that `search_pgvector.py` builds parameterised SQL `WHERE` conditions joined with `AND`. This test verifies the particular stage/trade combination shown; it is not an exhaustive test of all metadata combinations.

### Test 4 — Negative metadata filtering (no matching stage)

**Purpose:** Verify that an unmatched filter does not silently fall back to unrelated EKOs.

```bash
python -m backend.llm.rag_pipeline --question "Why is temporary wall bracing important?" --stage "NONEXISTENT_STAGE_TEST"
```

**Observed result:** `Retrieved EKOs: None`; `Cited sources: None`; `Answer status: insufficient_evidence`.

**Pass criteria:** No retrieved records and no invented answer or citation.

### Test 5 — Reject a fabricated EKO citation

**Purpose:** Unit-check that the citation validator rejects an answer citing an ID outside the allowed retrieved set.

```bash
python -c 'from backend.llm.rag_pipeline import validate_citations; validate_citations("Temporary bracing is required [MOCK-KO-999].", {"MOCK-KO-002"})'
```

**Observed result:** The validator raised:

```text
ValueError: Answer references EKOs not retrieved: MOCK-KO-999
```

**Pass criteria:** A `ValueError` is **expected** for this deliberately invalid citation; a nonzero exit status here indicates the rejection mechanism was triggered, not a pipeline defect. The test validates source-ID membership, **not** whether a claim is semantically entailed by its citation.

### Smoke-test outcome and interpretation

All **five manual smoke tests produced their expected outcomes** in the development environment. These observations complement, but are separate from, the automated 40-case synthetic benchmark below. Results may change if the corpus, embedding model, generation model, prompt, or metadata values change. The current tests do not establish correctness on approved SiteSeer EKOs or guarantee factual accuracy in real construction scenarios.

## Evaluation

There are **two distinct evaluation approaches**. Both currently use synthetic test data.

### 1. Fixed-evidence LLM comparison

The fixed-evidence benchmark gives each candidate model predefined question-and-evidence inputs. It evaluates generation behavior **without testing live pgvector retrieval**.

Benchmark dataset: `data/evaluation/rag_fixed_evidence_benchmark.json`

Run the local Qwen3 benchmark:

```bash
python -m backend.llm.run_rag_benchmark --model qwen3:4b-q4_K_M
```

The local Gemma3 comparison used the same benchmark with `--model gemma3:4b`. The OpenAI comparison runner is `backend/llm/run_openai_rag_benchmark.py`; it may incur API charges and requires separately configured access. **Do not run hosted-model benchmarks without authorization.**

Recorded development comparison:

| Metric | Qwen3 4B | Gemma3 4B | GPT-4.1 mini |
| --- | ---: | ---: | ---: |
| Expected citation present (answerable cases) | 35/35 | 35/35 | 35/35 |
| Cases with unsupported citation | 0 | 5 | 0 |
| Insufficient-evidence keyword/refusal check | 5/5 | 5/5 | 5/5 |
| Mean latency (seconds) | 5.923 | 5.149 | 1.615 |

These are **development benchmark observations**, not general model rankings. Hosted-model latency and cost depend on service and usage conditions. Qwen3 was selected provisionally for the local-first prototype, considering grounding checks and the absence of hosted inference charges.

Saved results include:

- `data/evaluation/results/ollama_qwen3_4b-q4_K_M.json`
- `data/evaluation/results/ollama_gemma3_4b.json`
- `data/evaluation/results/openai_gpt-4.1-mini.json`
- `data/evaluation/results/rag_model_comparison.json`

### 2. End-to-end retrieval and generation

This evaluation runs the complete path: question embedding, pgvector retrieval, evidence construction, Ollama generation, status parsing, and citation validation.

```bash
python -m backend.llm.evaluate_end_to_end_rag --output data/evaluation/results/end_to_end_qwen3_4b_prompt_v2.json
```

**Note:** This command writes to the named file. Use a new output filename if you want to preserve the committed benchmark unchanged.

Verified results for the updated prompt, on **40 synthetic cases**:

| Metric | Result |
| --- | ---: |
| Answerable cases | 35 |
| Insufficient-evidence cases | 5 |
| Retrieval Hit@1 | 35/35 answerable cases |
| Retrieval Hit@3 | 35/35 answerable cases |
| Answerable cases with citations | 35/35 |
| Answerable cases answered | 35/35 |
| Insufficient-evidence cases refused | 5/5 |
| Citation validation failures | 0 |
| Invalid model responses | 0 |
| Cases with citations outside retrieved EKOs | 0 |
| Mean per-case latency | 5.783 seconds |

Reference report: `data/evaluation/results/end_to_end_qwen3_4b_prompt_v2.json`.

Earlier runs, targeted insufficient-evidence tests, and smoke-test outputs are retained under `data/evaluation/results/` for traceability. The benchmark checks retrieval, status, and citation-format behavior; it is **not a comprehensive independent factual-grounding audit**.

### Interpreting results responsibly

- The 40 cases are synthetic and relatively small; they do not establish performance on real SiteSeer content or unseen construction scenarios.
- Citation validation checks EKO identifiers and formatting, **not claim-level entailment**.
- The five insufficient-evidence cases test specific examples; they do not establish reliable refusal behavior for every unsupported question.
- Similarity scores are retrieval-ranking signals, not confidence probabilities.
- Latency is machine- and environment-dependent.
- Model and retrieval choices remain provisional until evaluated on representative, approved data.

## Source Governance and Data Boundaries

- The current index is derived from **synthetic search-ready EKOs**. Mock IDs such as `MOCK-KO-037` are not authoritative client knowledge identifiers.
- The intended integration boundary is approved, eligible EKO content supplied through the Group 2A knowledge workflow; Group 2A remains responsible for its authoritative knowledge and approval processes.
- Raw evidence records (EVDs) must not be indexed directly as if they were approved educational knowledge.
- Current mock-data eligibility handling must not be presented as proof that the real client-data governance integration is complete.
- Do not commit credentials, confidential client records, or unapproved client content to GitHub.

## Main Files

| File | Purpose |
| --- | --- |
| `backend/llm/rag_pipeline.py` | Retrieve evidence, call Qwen3, parse response, validate citations, and expose the CLI. |
| `backend/llm/run_rag_benchmark.py` | Run local Ollama fixed-evidence benchmarks; contains the Ollama HTTP client. |
| `backend/llm/run_openai_rag_benchmark.py` | Hosted-model comparison runner. |
| `backend/llm/evaluate_rag_benchmark.py` | Evaluate fixed-evidence model results. |
| `backend/llm/evaluate_end_to_end_rag.py` | Evaluate full retrieval-plus-generation behavior. |
| `backend/llm/test_run_rag_benchmark.py` | Tests for benchmark runner behavior. |
| `backend/llm/test_evaluate_rag_benchmark.py` | Tests for evaluation behavior. |
| `backend/semantic_search/search_pgvector.py` | Embedding-based retrieval and metadata filters. |

## Troubleshooting

**Ollama connection refused:** Start the Ollama application/service and check that the local API is reachable on port `11434`.

**Model not found:** Run `ollama list`, then `ollama pull qwen3:4b-q4_K_M` if needed.

**Database connection error or missing `knowledge_embeddings` table:** Follow the semantic search README to configure the local PostgreSQL connection and rebuild the index. GitHub does not contain the local database.

**Unexpected `insufficient_evidence` response:** Check whether the indexed corpus contains relevant, eligible content and whether metadata filters are overly restrictive. Do not treat high similarity alone as proof of answerability.

**Citation validation fallback:** Inspect the retrieved IDs and generated answer. A valid EKO citation must match an ID in the retrieved evidence; changing the validator to accept invented IDs is not an appropriate fix.

**Hugging Face unauthenticated-request warning:** Public model downloads may still work without an access token; the warning concerns access limits and download performance, not necessarily a failed pipeline run.