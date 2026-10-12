# SiteSeer 2B : AI Query Layer - Knowledge Corpus Intelligence

## Project Overview

SiteSeer Project 2B develops an AI-assisted query layer that enables learners to ask construction-related questions and receive answers grounded in SiteSeer knowledge.

The prototype uses semantic search and Retrieval-Augmented Generation (RAG) to retrieve relevant Educational Knowledge Objects (EKOs), generate responses, and provide traceable knowledge-object citations.

**Current scope:** A working local prototype evaluated with synthetic EKO data. Integration with approved client EKOs remains pending.

## System Architecture

```text
Learner question
      ↓
MiniLM question embedding
      ↓
PostgreSQL + pgvector retrieval
      ↓
Top-K search-ready EKOs
      ↓
Local Qwen3 model via Ollama
      ↓
Structured answer or insufficient-evidence response
      ↓
Citation and response validation
```

## Technology Stack

| Component | Technology |
| --- | --- |
| Language | Python |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Vector database | PostgreSQL + pgvector |
| Retrieval | Cosine similarity, optional metadata filters |
| Generation model | `qwen3:4b-q4_K_M` |
| Local LLM runtime | Ollama |

## Documentation

- [Semantic Search](backend/semantic_search/README.md) — environment setup, embedding generation, pgvector indexing, semantic retrieval, and retrieval evaluation.
- [RAG Pipeline](backend/llm/README.md) — local model setup, end-to-end question answering, citation validation, model comparison, and RAG evaluation.

Start with the semantic search guide to configure the local database and index. Then follow the RAG guide to run the complete pipeline.

## Development Status

The following components have been implemented using synthetic development data:

- Search-ready EKO preprocessing and embedding generation
- PostgreSQL + pgvector indexing and semantic retrieval
- Optional stage, trade, jurisdiction, and learner-level filters
- Local LLM comparison and provisional Qwen3 selection
- End-to-end RAG answer generation
- Structured response and EKO citation validation
- Insufficient-evidence response handling
- Synthetic benchmark evaluation

### Latest end-to-end synthetic benchmark

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
