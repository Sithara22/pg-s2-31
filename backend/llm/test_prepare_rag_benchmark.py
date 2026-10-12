"""Tests for fixed-evidence benchmark preparation."""
import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name('prepare_rag_benchmark.py')
spec = importlib.util.spec_from_file_location('prepare_rag_benchmark', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BenchmarkPreparationTests(unittest.TestCase):
    def setUp(self):
        self.ekos = mod.read_array(mod.EKO_PATH)
        self.tests = mod.read_array(mod.TEST_PATH)

    def test_expected_case_counts(self):
        cases = mod.prepare(self.ekos, self.tests)
        self.assertEqual(len(cases), 40)
        self.assertEqual(sum(c['scenario'] == 'answerable_candidate' for c in cases), 35)
        self.assertEqual(sum(c['scenario'] == 'insufficient_zero_evidence' for c in cases), 5)

    def test_no_expected_answer_label_in_model_input(self):
        cases = mod.prepare(self.ekos, self.tests)
        for c in cases:
            self.assertNotIn('expected_ko_id', c['model_input'])
            self.assertNotIn('scenario', c['model_input'])

    def test_excluded_eko_not_eligible(self):
        excluded = next(e for e in self.ekos if not mod.is_mock_eligible(e))
        self.assertFalse(mod.is_mock_eligible(excluded))

    def test_answerable_evidence_is_expected_ko(self):
        cases = mod.prepare(self.ekos, self.tests)
        for c in cases:
            if c['expected_ko_id'] is not None:
                self.assertEqual(c['evidence_ko_ids'], [c['expected_ko_id']])
            else:
                self.assertEqual(c['evidence_ko_ids'], [])

    def test_filter_mismatch_fails(self):
        test = dict(self.tests[0], filters={'jurisdiction': 'VIC'})
        with self.assertRaises(ValueError):
            mod.prepare(self.ekos, [test])


if __name__ == '__main__':
    unittest.main()
