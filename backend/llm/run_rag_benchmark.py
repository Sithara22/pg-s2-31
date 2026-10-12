"""Run fixed-evidence SiteSeer RAG benchmarks against a local Ollama model."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = ROOT / 'data/evaluation/rag_fixed_evidence_benchmark.json'
RESULTS_DIR = ROOT / 'data/evaluation/results'
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'


def load_benchmark(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Benchmark must contain a non-empty JSON array.')
    ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Benchmark records must be JSON objects.')
        test_id = record.get('test_id')
        model_input = record.get('model_input')
        if not isinstance(test_id, str) or not test_id or test_id in ids:
            raise ValueError('Benchmark test IDs must be unique non-empty strings.')
        ids.add(test_id)
        if not isinstance(model_input, dict) or any(
            not isinstance(model_input.get(key), str) or not model_input[key].strip()
            for key in ('system', 'user')
        ):
            raise ValueError(f'{test_id}: invalid model input.')
        if record.get('scenario') not in ('answerable_candidate', 'insufficient_zero_evidence'):
            raise ValueError(f'{test_id}: invalid scenario.')
        if not isinstance(record.get('evidence_ko_ids'), list):
            raise ValueError(f'{test_id}: invalid evidence KO IDs.')
    return records


def query_ollama(model: str, model_input: dict, timeout: float) -> dict:
    payload = {
        'model': model,
        'stream': False,
        'think': False,
        'messages': [
            {'role': 'system', 'content': model_input['system']},
            {'role': 'user', 'content': model_input['user']},
        ],
        'options': {'temperature': 0, 'num_predict': 512},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f'Ollama request failed: {exc}') from exc
    message = data.get('message')
    if not isinstance(message, dict) or not isinstance(message.get('content'), str):
        raise ValueError('Ollama returned an invalid response.')
    answer = message['content'].strip()
    if not answer or '<think' in answer.lower() or '</think>' in answer.lower() or message.get('thinking'):
        raise ValueError('Ollama response contains reasoning output or no final answer.')
    return {
        'answer': answer,
        'input_tokens': data.get('prompt_eval_count'),
        'output_tokens': data.get('eval_count'),
    }


def run(records: list[dict], model: str, timeout: float) -> list[dict]:
    results = []
    for record in records:
        start = time.perf_counter()
        output = query_ollama(model, record['model_input'], timeout)
        results.append({
            'test_id': record['test_id'],
            'scenario': record['scenario'],
            'expected_ko_id': record.get('expected_ko_id'),
            'evidence_ko_ids': record['evidence_ko_ids'],
            'model': model,
            'provider': 'ollama',
            'answer': output['answer'],
            'latency_seconds': round(time.perf_counter() - start, 3),
            'input_tokens': output['input_tokens'],
            'output_tokens': output['output_tokens'],
            'estimated_api_cost_usd': '0',
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Installed Ollama model name')
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK_PATH)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    records = load_benchmark(args.benchmark)
    print('SiteSeer 2B | Local RAG Benchmark')
    print('--------------------------------')
    print(f'Model:       {args.model}')
    print(f'Test cases:  {len(records)}')
    if args.validate_only:
        print('Status:      Validated')
        return
    results = run(records, args.model, args.timeout)
    safe_model = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in args.model)
    output_path = args.output or RESULTS_DIR / f'ollama_{safe_model}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'benchmark_type': 'fixed_evidence_synthetic',
        'provider': 'ollama',
        'model': args.model,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'generation_options': {'temperature': 0, 'num_predict': 512, 'think': False},
        'results': results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    try:
        label = output_path.relative_to(ROOT)
    except ValueError:
        label = output_path
    print(f'Completed:   {len(results)}')
    print(f'Output:      {label}')
    print('Status:      Completed')


if __name__ == '__main__':
    main()
