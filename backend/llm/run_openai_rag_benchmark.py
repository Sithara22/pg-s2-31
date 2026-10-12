"""Fixed-evidence GPT-4.1 mini benchmark; offline by default, paid only with --live."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from openai_budget_test import (
    BUDGET, INPUT_RATE, OUTPUT_RATE, RESERVE, MODEL, LEDGER,
    read_ledger, reserve_call, record_success, write_ledger,
)
from run_rag_benchmark import BENCHMARK_PATH, RESULTS_DIR, load_benchmark

MAX_OUTPUT_TOKENS = 512
OUTPUT_PATH = RESULTS_DIR / 'openai_gpt-4.1-mini.json'


def estimated_upper_bound(records: list[dict]) -> Decimal:
    """Illustrative estimate only: assumes <= 2 input tokens per character."""
    chars = sum(len(r['model_input']['system']) + len(r['model_input']['user']) for r in records)
    return (Decimal(chars * 2) * INPUT_RATE
            + Decimal(len(records) * MAX_OUTPUT_TOKENS) * OUTPUT_RATE)


def run_live(records: list[dict], output_path: Path, timeout: float) -> None:
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is not set; do not paste credentials into code.')
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=timeout)
    results = []
    # Stop on errors; retain reservations for uncertain charges. Save each completed result.
    for record in records:
        if Decimal(read_ledger()['spent_usd']) + RESERVE > BUDGET:
            raise RuntimeError('Local budget reservation limit reached; stopping before next call.')
        index = reserve_call()
        start = time.perf_counter()
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=record['model_input']['system'],
                input=record['model_input']['user'],
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )
            if response.status != 'completed' or not response.output_text.strip() or response.usage is None:
                raise ValueError('Incomplete response or missing usage; stopping to avoid unreliable results.')
            cost = record_success(index, response)
        except Exception:
            ledger = read_ledger()
            ledger['entries'][index]['status'] = 'uncertain_or_failed_keep_reservation'
            write_ledger(ledger)
            raise
        results.append({
            'test_id': record['test_id'],
            'scenario': record['scenario'],
            'expected_ko_id': record.get('expected_ko_id'),
            'evidence_ko_ids': record['evidence_ko_ids'],
            'model': MODEL,
            'provider': 'openai',
            'answer': response.output_text.strip(),
            'latency_seconds': round(time.perf_counter() - start, 3),
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens,
            'estimated_api_cost_usd': str(cost),
        })
        report = {
            'benchmark_type': 'fixed_evidence_synthetic',
            'provider': 'openai',
            'model': MODEL,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'generation_options': {'temperature': 0, 'max_output_tokens': MAX_OUTPUT_TOKENS, 'store': False},
            'results': results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temporary.replace(output_path)
        print(f"Completed {len(results)}/{len(records)}: {record['test_id']}; estimated cost ${cost}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=BENCHMARK_PATH)
    parser.add_argument('--output', type=Path, default=OUTPUT_PATH)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--live', action='store_true', help='Make paid API requests; requires separate explicit approval')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    records = load_benchmark(args.benchmark)
    spent = Decimal(read_ledger()['spent_usd'])
    print(f'Model: {MODEL}; test cases: {len(records)}; local ledger: ${spent} / ${BUDGET}')
    print(f'Illustrative total cost estimate (not a cap): ${estimated_upper_bound(records):.6f}')
    print(f'Per-call reservation: ${RESERVE} (not a hard API spending cap)')
    if not args.live:
        print('OFFLINE VALIDATION PASSED: no API requests or file writes.')
        return
    if args.output.exists():
        raise RuntimeError(f'Output already exists: {args.output}. Refusing to overwrite paid results.')
    run_live(records, args.output, args.timeout)
    print(f'Benchmark completed: {args.output}')


if __name__ == '__main__':
    main()
