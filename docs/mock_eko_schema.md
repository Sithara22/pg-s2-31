# Mock EKO Schema

## Purpose

This document defines the temporary Educational Knowledge Object (EKO) structure
used by Project 2B while the Project 2A Knowledge Query Contract is unavailable.

This is a development schema only. It is NOT the authoritative SiteSeer EKO
schema.

When the Project 2A query contract becomes available, this schema and the
associated data-loading code must be reviewed and updated.

## Fields

| Field | Category | Description | Intended 2B Use |
|---|---|---|---|
| ko_id | Identity | Unique Knowledge Object identifier | Identification and citation |
| title | Searchable | Learner-facing knowledge title | Embedding and display |
| l1 | Searchable | Factual knowledge | Embedding |
| l2 | Searchable | Procedural knowledge | Embedding |
| l3 | Searchable | Conditional knowledge | Embedding |
| l4 | Searchable | Expert judgement | Embedding |
| l5 | Searchable | Context/dependency knowledge | Embedding |
| stage | Metadata | Construction stage | Filtering |
| trade | Metadata | Relevant trade/domain | Filtering |
| jurisdiction | Metadata | Applicable jurisdiction | Filtering |
| learner_level | Metadata | Intended learner level | Response/retrieval context |
| publication_status | Governance | Publication state | Index eligibility |
| validation_status | Governance | Validation state | Index eligibility |
| ai_indexable | Governance | Whether record is approved for AI indexing | Index eligibility |
| superseded | Governance | Whether knowledge has been superseded | Index eligibility |
| version | Governance | Record version | Traceability/re-indexing |

## Initial Indexing Assumption

For the mock implementation, a record will only be considered indexable when:

- publication_status is `published`
- validation_status is `validated`
- ai_indexable is `true`
- superseded is `false`

These rules are temporary development assumptions and must be replaced or
confirmed when Project 2A provides the authoritative query contract.

## Searchable Text

The initial whole-object semantic-search baseline will construct searchable text
from:

- title
- L1
- L2
- L3
- L4
- L5

This does not imply that whole-object embedding is the final retrieval
architecture. Field-level and hybrid retrieval will be evaluated later.