import unittest

from evaluate_rag_benchmark import evaluate


class EvaluateRagBenchmarkTests(unittest.TestCase):
    def report(self, answer, scenario="insufficient_zero_evidence", evidence=None, expected=None):
        return {"model": "test", "benchmark_type": "fixed_evidence_synthetic", "results": [{
            "test_id": "RET-001", "scenario": scenario, "evidence_ko_ids": evidence or [],
            "expected_ko_id": expected, "answer": answer, "latency_seconds": 1.0,
            "input_tokens": 10, "output_tokens": 5, "estimated_api_cost_usd": "0",
        }]}

    def test_false_citation_in_refusal(self):
        result = evaluate(self.report("Insufficient evidence [MOCK-KO-001]."))
        self.assertEqual(result["zero_evidence_unsupported_citation_cases"], 1)
        self.assertEqual(result["zero_evidence_refusals_detected"], 1)

    def test_clean_refusal(self):
        result = evaluate(self.report("Available evidence is insufficient."))
        self.assertEqual(result["cases_with_unsupported_citations"], 0)

    def test_supported_citation(self):
        result = evaluate(self.report("Answer [MOCK-KO-002]", "answerable_candidate", ["MOCK-KO-002"], "MOCK-KO-002"))
        self.assertEqual(result["expected_citation_present"], 1)
        self.assertEqual(result["cases_with_unsupported_citations"], 0)

    def test_unsupported_answerable_citation(self):
        result = evaluate(self.report("Answer [MOCK-KO-999]", "answerable_candidate", ["MOCK-KO-002"], "MOCK-KO-002"))
        self.assertEqual(result["expected_citation_present"], 0)
        self.assertEqual(result["cases_with_unsupported_citations"], 1)

    def test_clean_refusal_does_not_require_review(self):
        detail = evaluate(self.report("Available evidence is insufficient."))["case_details"][0]
        self.assertFalse(detail["manual_review_required"])
        self.assertEqual(detail["review_reasons"], [])

    def test_false_citation_requires_review(self):
        detail = evaluate(self.report("Insufficient evidence [MOCK-KO-001]."))["case_details"][0]
        self.assertTrue(detail["manual_review_required"])
        self.assertIn("unsupported_ko_citation", detail["review_reasons"])

    def test_answerable_case_requires_claim_review(self):
        detail = evaluate(self.report("Answer [MOCK-KO-002]", "answerable_candidate", ["MOCK-KO-002"], "MOCK-KO-002"))["case_details"][0]
        self.assertEqual(detail["review_reasons"], ["human_claim_level_grounding_review"])

    def test_missing_citation_requires_review(self):
        detail = evaluate(self.report("Answer", "answerable_candidate", ["MOCK-KO-002"], "MOCK-KO-002"))["case_details"][0]
        self.assertIn("missing_expected_ko_citation", detail["review_reasons"])

    def test_unexpected_refusal_requires_review(self):
        detail = evaluate(self.report("Insufficient evidence [MOCK-KO-002]", "answerable_candidate", ["MOCK-KO-002"], "MOCK-KO-002"))["case_details"][0]
        self.assertIn("unexpected_refusal_in_answerable_case", detail["review_reasons"])

    def test_missing_zero_evidence_refusal_requires_review(self):
        detail = evaluate(self.report("Here is a definite answer."))["case_details"][0]
        self.assertIn("missing_refusal_in_zero_evidence_case", detail["review_reasons"])

    def test_duplicate_ids_rejected(self):
        report = self.report("Insufficient")
        report["results"].append(report["results"][0].copy())
        with self.assertRaises(ValueError):
            evaluate(report)

    def test_inconsistent_zero_evidence_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.report("Insufficient", evidence=["MOCK-KO-001"]))


if __name__ == "__main__":
    unittest.main()
