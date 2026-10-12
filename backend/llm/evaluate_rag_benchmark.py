import argparse
import json
import re
import statistics
from decimal import Decimal
from pathlib import Path

KO_PATTERN = re.compile(r"\bMOCK-KO-\d+\b")
REFUSAL_PATTERN = re.compile(r"\b(?:insufficient|cannot answer|can't answer|unable to answer|not enough (?:evidence|information))\b", re.I)


def evaluate(report):
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Report must contain non-empty results")
    ids = [r["test_id"] for r in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test IDs")
    details = []
    for row in results:
        evidence = set(row["evidence_ko_ids"])
        cited = set(KO_PATTERN.findall(row["answer"]))
        expected = row["expected_ko_id"]
        if row["scenario"] not in ("answerable_candidate", "insufficient_zero_evidence"):
            raise ValueError(f"Unsupported scenario: {row['scenario']}")
        if row["scenario"] == "insufficient_zero_evidence" and (evidence or expected is not None):
            raise ValueError(f"Inconsistent zero-evidence case: {row['test_id']}")
        is_answerable = row["scenario"] == "answerable_candidate"
        expected_present = bool(expected and expected in cited)
        unsupported = sorted(cited - evidence)
        refusal = bool(REFUSAL_PATTERN.search(row["answer"]))
        reasons = []
        if unsupported:
            reasons.append("unsupported_ko_citation")
        if is_answerable and not expected_present:
            reasons.append("missing_expected_ko_citation")
        if is_answerable and refusal:
            reasons.append("unexpected_refusal_in_answerable_case")
        if not is_answerable and not refusal:
            reasons.append("missing_refusal_in_zero_evidence_case")
        if is_answerable:
            reasons.append("human_claim_level_grounding_review")
        details.append({
            "test_id": row["test_id"], "scenario": row["scenario"],
            "expected_citation_present": expected_present,
            "unsupported_citations": unsupported,
            "refusal_detected": refusal,
            "manual_review_required": bool(reasons),
            "review_reasons": reasons,
            "latency_seconds": row["latency_seconds"],
        })
    answerable = [d for d in details if d["scenario"] == "answerable_candidate"]
    zero = [d for d in details if d["scenario"] == "insufficient_zero_evidence"]
    latencies = [d["latency_seconds"] for d in details]
    return {
        "model": report["model"], "benchmark_type": report["benchmark_type"],
        "cases": len(details), "answerable_cases": len(answerable), "zero_evidence_cases": len(zero),
        "expected_citation_present": sum(d["expected_citation_present"] for d in answerable),
        "cases_with_unsupported_citations": sum(bool(d["unsupported_citations"]) for d in details),
        "zero_evidence_unsupported_citation_cases": sum(bool(d["unsupported_citations"]) for d in zero),
        "zero_evidence_refusals_detected": sum(d["refusal_detected"] for d in zero),
        "manual_review_required_cases": sum(d["manual_review_required"] for d in details),
        "mean_latency_seconds": round(statistics.mean(latencies), 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "input_tokens": sum(r["input_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "estimated_api_cost_usd": str(sum((Decimal(str(r["estimated_api_cost_usd"])) for r in results), Decimal("0"))),
        "case_details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed-evidence SiteSeer RAG reports offline")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [evaluate(json.loads(path.read_text(encoding="utf-8"))) for path in args.reports]
    if len(summaries) > 1:
        first = {(d["test_id"], d["scenario"]) for d in summaries[0]["case_details"]}
        if any({(d["test_id"], d["scenario"]) for d in s["case_details"]} != first for s in summaries[1:]):
            raise ValueError("Reports do not contain matching test cases")
        if len({s["benchmark_type"] for s in summaries}) != 1:
            raise ValueError("Reports use different benchmark types")
    result = {"evaluations": summaries, "note": "Citation IDs and refusal wording are heuristic checks, not factual groundedness judgments."}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
