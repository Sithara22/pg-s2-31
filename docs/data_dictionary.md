# SiteSeer 2B Data Dictionary

## Purpose

This document records the SiteSeer knowledge structures relevant to Project 2B
(AI Query Layer — Knowledge Corpus Intelligence).

Project 2B consumes approved knowledge from Project 2A and uses it for semantic
retrieval and Retrieval-Augmented Generation (RAG).

The final data structure and field names will depend on the Knowledge Query
Contract provided by Project 2A. Until that contract is available, Project 2B
will use mock EKO records for development.

---

## 1. SiteSeer Knowledge Layers

### Evidence Record (EVD)

EVDs contain raw practitioner evidence captured from expert interviews.

Purpose:
- Preserve source evidence and practitioner statements.
- Support provenance of later knowledge records.

Use in Project 2B:
- Raw EVDs must NOT be included in the learner-facing semantic search index.
- EVD content will not be used as normal embedding input.

---

### Construction Decision Object (CDO)

CDOs structure practitioner reasoning around construction decisions.

Examples of information represented by a CDO include:
- trigger
- preconditions
- inputs
- alternatives
- selection rule
- conditions
- risks
- evidence of completion
- failure mode
- consequence chain
- relationships
- provenance and validation information

Use in Project 2B:
- Project 2B must not assume that all CDO content is indexable.
- Only CDO fields explicitly approved for AI indexing may be used.
- Project 2A controls publication, validation and accessibility.

---

### Educational Knowledge Object (EKO)

EKOs contain learner-facing knowledge derived from validated construction
knowledge.

Published EKOs are the primary knowledge objects used by Project 2B for
semantic search and RAG.

The learner-facing knowledge is organised into L1-L5.

| Level | Meaning | Retrieval Purpose |
|---|---|---|
| L1 | Factual knowledge | Questions about what something is |
| L2 | Procedural knowledge | Questions about how something is done |
| L3 | Conditional knowledge | Questions about when conditions or rules change |
| L4 | Expert judgement | Questions involving practitioner judgement or observations |
| L5 | Context and dependencies | Questions about dependencies, sequence and relationships |

---

## 2. Project 2B Data Categories

Project 2B separates EKO information into three broad categories.

### Searchable Content

Text that may be used to generate embeddings for semantic retrieval.

Potential fields:
- title
- L1
- L2
- L3
- L4
- L5

The exact fields will be confirmed by the Project 2A query contract.

### Metadata

Structured information that can support filtering, ranking, context and
traceability.

Potential examples:
- Knowledge Object ID
- construction stage
- trade/domain
- jurisdiction
- learner level
- relationships

The exact metadata fields will be confirmed by Project 2A.

### Governance Information

Information used to determine whether a record is permitted to enter the
learner-facing AI index.

Potential examples:
- publication status
- validation status
- current/superseded status
- standards verification status
- AI-indexable/approval status
- version/schema version

The exact governance fields and accepted values must be supplied or confirmed
by Project 2A.

---

## 3. Publication Boundary

The Project 2B learner index must only contain content approved for AI indexing.

According to the Project 2B scope, the learner index may include:
- published EKOs
- validated and explicitly approved CDO fields
- current records
- records applicable to the required jurisdiction
- records with acceptable standards-verification status

The learner index must exclude:
- raw EVDs
- draft CDOs
- personally identifying expert information
- superseded knowledge
- confidential project documents
- unverified standards claims

Project 2A remains the authoritative source of knowledge.

The Project 2B vector index is a derived and rebuildable representation of
approved SiteSeer knowledge.

---

## 4. Temporary Development Data

The Project 2A Knowledge Query Contract is not currently available.

To avoid blocking development, Project 2B will use mock EKO records during the
initial implementation.

The mock schema is temporary and must not be treated as the authoritative
SiteSeer EKO schema.

When the Project 2A query contract becomes available:

1. The mock schema will be compared with the agreed contract.
2. Field mappings will be updated.
3. The data loader will be adapted to consume Project 2A data.
4. Publication and validation rules will be updated to match the contract.
5. The semantic index will be rebuilt from approved Project 2A content.