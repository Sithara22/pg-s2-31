"""Unit tests for the local fixed-evidence RAG benchmark runner."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name('run_rag_benchmark.py')
SPEC = importlib.util.spec_from_file_location('run_rag_benchmark', MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            'test_id': 'RET-001',
            'scenario': 'answerable_candidate',
            'expected_ko_id': 'MOCK-KO-001',
            'evidence_ko_ids': ['MOCK-KO-001'],
            'model_input': {'system': 'Use only evidence.', 'user': 'Question and evidence'},
        }

    def test_load_valid_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            path.write_text(json.dumps([self.record]), encoding='utf-8')
            self.assertEqual(len(runner.load_benchmark(path)), 1)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            path.write_text(json.dumps([self.record, self.record]), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_benchmark(path)

    def test_missing_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            record = {**self.record, 'model_input': {'system': '', 'user': 'Question'}}
            path.write_text(json.dumps([record]), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_benchmark(path)

    @patch.object(runner, 'query_ollama', return_value={
        'answer': 'Evidence-based answer [MOCK-KO-001]',
        'input_tokens': 100,
        'output_tokens': 20,
    })
    def test_run_records_response_and_metadata(self, mock_query):
        results = runner.run([self.record], 'local-test-model', 30)
        self.assertEqual(results[0]['test_id'], 'RET-001')
        self.assertEqual(results[0]['input_tokens'], 100)
        self.assertEqual(results[0]['estimated_api_cost_usd'], '0')
        mock_query.assert_called_once_with('local-test-model', self.record['model_input'], 30)

    @patch.object(runner, 'urlopen')
    def test_request_disables_thinking(self, mock_urlopen):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            'message': {'role': 'assistant', 'content': 'READY'},
            'prompt_eval_count': 10,
            'eval_count': 2,
        }).encode('utf-8')
        mock_urlopen.return_value = response
        result = runner.query_ollama('qwen3:4b-q4_K_M', self.record['model_input'], 30)
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertIs(payload['think'], False)
        self.assertEqual(result['answer'], 'READY')

    @patch.object(runner, 'urlopen')
    def test_reasoning_in_content_rejected(self, mock_urlopen):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            'message': {'role': 'assistant', 'content': 'Reasoning...</think>\nREADY'},
        }).encode('utf-8')
        mock_urlopen.return_value = response
        with self.assertRaisesRegex(ValueError, 'reasoning'):
            runner.query_ollama('qwen3:4b', self.record['model_input'], 30)

    def test_invalid_scenario_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            record = {**self.record, 'scenario': 'unknown'}
            path.write_text(json.dumps([record]), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.load_benchmark(path)


if __name__ == '__main__':
    unittest.main()
