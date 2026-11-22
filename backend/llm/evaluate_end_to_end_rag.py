
"""Evaluate SiteSeer retrieval and local Qwen3 generation."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import SentenceTransformer
from backend.llm.rag_pipeline import answer_question, OLLAMA_MODEL as LLM_MODEL
from backend.semantic_search.search_pgvector import (
    MODEL_NAME,
    get_database_connection,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data/evaluation/rag_fixed_evidence_benchmark.json"
DEFAULT_OUTPUT = ROOT / "data/evaluation/results/end_to_end_qwen3_4b.json"

FILTER_KEYS = {"stage", "trade", "jurisdiction", "learner_level"}
VALID_STATUSES = {
    "conversational",
    "answered",
    "insufficient_evidence",
    "citation_validation_failed",
    "invalid_model_response",
}

CITATION_PATTERN = re.compile(r"\[(MOCK-KO-\d+)\]")


def load_cases(path: Path) -> list[dict]:
    """Load and validate synthetic benchmark cases."""

    cases = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark must be a non-empty JSON array.")

    seen = set()

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object.")

        case_id = case.get("test_id")

        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
        ):
            raise ValueError(
                "Test IDs must be unique non-empty strings."
            )

        seen.add(case_id)

        if (
            not isinstance(case.get("question"), str)
            or not case["question"].strip()
        ):
            raise ValueError(f"{case_id}: missing question.")

        if case.get("scenario") not in {
            "answerable_candidate",
            "insufficient_zero_evidence",
        }:
            raise ValueError(
                f"{case_id}: unsupported scenario."
            )

        filters = case.get("filters", {})

        if not isinstance(filters, dict) or set(filters) - FILTER_KEYS:
            raise ValueError(
                f"{case_id}: unsupported metadata filters."
            )

        if any(
            not isinstance(value, str)
            for value in filters.values()
        ):
            raise ValueError(
                f"{case_id}: filter values must be strings."
            )

        if (
            case["scenario"] == "answerable_candidate"
            and not case.get("expected_ko_id")
        ):
            raise ValueError(
                f"{case_id}: missing expected KO ID."
            )

    return cases


def evaluate(cases: list[dict], top_k: int) -> dict:
    """Run retrieval and generation for each benchmark question."""

    model = SentenceTransformer(MODEL_NAME)
    results = []

    connection = get_database_connection()

    try:
        for index, case in enumerate(cases, 1):
            start = time.perf_counter()

            outcome = answer_question(
                question=case["question"],
                embedding_model=model,
                connection=connection,
                top_k=top_k,
                **case.get("filters", {}),
            )

            elapsed = round(
                time.perf_counter() - start,
                3,
            )

            status = outcome.get("status")

            if status not in VALID_STATUSES:
                raise ValueError(
                    f"{case['test_id']}: invalid pipeline status "
                    f"{status!r}"
                )

            retrieved = [
                item["ko_id"]
                for item in outcome["retrieval"]
            ]

            cited = CITATION_PATTERN.findall(
                outcome["answer"]
            )

            unsupported = sorted(
                set(cited) - set(retrieved)
            )

            expected = case.get("expected_ko_id")

            result = {
                "test_id": case["test_id"],
                "scenario": case["scenario"],
                "question": case["question"],
                "filters": case.get("filters", {}),
                "expected_ko_id": expected,
                "retrieved_ko_ids": retrieved,
                "retrieval": outcome["retrieval"],
                "hit_at_1": (
                    expected == retrieved[0]
                    if expected and retrieved
                    else False
                ),
                "hit_at_k": (
                    expected in retrieved
                    if expected
                    else False
                ),
                "answer": outcome["answer"],
                "answer_status": status,
                "cited_ko_ids": cited,
                "unsupported_citations": unsupported,
                "latency_seconds": elapsed,
            }

            results.append(result)

            print(
                f"[{index}/{len(cases)}] "
                f"{case['test_id']} | "
                f"{elapsed:.2f}s | "
                f"hit@{top_k}={result['hit_at_k']} | "
                f"status={status}",
                flush=True,
            )

    finally:
        connection.close()

    answerable = [
        r for r in results
        if r["scenario"] == "answerable_candidate"
    ]

    insufficient = [
        r for r in results
        if r["scenario"] == "insufficient_zero_evidence"
    ]

    status_counts = Counter(
        r["answer_status"]
        for r in results
    )

    summary = {
        "total_cases": len(results),
        "answerable_cases": len(answerable),
        "insufficient_cases": len(insufficient),
        "hit_at_1": sum(
            r["hit_at_1"]
            for r in answerable
        ),
        "hit_at_k": sum(
            r["hit_at_k"]
            for r in answerable
        ),
        "top_k": top_k,
        "answerable_cases_with_citation": sum(
            bool(r["cited_ko_ids"])
            for r in answerable
        ),
        "answerable_cases_answered": sum(
            r["answer_status"] == "answered"
            for r in answerable
        ),
        "insufficient_cases_refused": sum(
            r["answer_status"] == "insufficient_evidence"
            for r in insufficient
        ),
        "citation_validation_failures": status_counts[
            "citation_validation_failed"
        ],
        "invalid_model_responses": status_counts[
            "invalid_model_response"
        ],
        "status_counts": dict(status_counts),
        "cases_with_unsupported_citations": sum(
            bool(r["unsupported_citations"])
            for r in results
        ),
        "mean_latency_seconds": round(
            sum(
                r["latency_seconds"]
                for r in results
            ) / len(results),
            3,
        ),
    }

    return {
        "benchmark_type": (
            "end_to_end_synthetic_retrieval_generation"
        ),
        "model": "Qwen3-4B",
        "embedding_model": "all-MiniLM-L6-v2",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "notes": (
            "Retrieval metrics are based on synthetic expected EKO IDs. "
            "Citation checks validate references, not claim-level grounding. "
            "Answer statuses are reported by the RAG pipeline and are not "
            "independent verification of factual correctness. "
            "Insufficient-evidence cases may retrieve unrelated EKOs."
        ),
        "summary": summary,
        "results": results,
    }


def main() -> None:
    """Run the end-to-end evaluation from the command line."""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases for a smoke test",
    )

    args = parser.parse_args()

    if (
        args.top_k < 1
        or (
            args.limit is not None
            and args.limit < 1
        )
    ):
        parser.error(
            "--top-k and --limit must be positive"
        )

    cases = load_cases(args.benchmark)

    if args.limit is not None:
        cases = cases[:args.limit]

    report = evaluate(
        cases,
        args.top_k,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nSummary:")
    print(
        json.dumps(
            report["summary"],
            indent=2,
        )
    )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
