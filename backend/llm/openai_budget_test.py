"""SiteSeer 2B: explicitly opted-in OpenAI smoke test with local budget ledger."""
import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

BUDGET = Decimal('1.00')
WARNING = Decimal('0.75')
RESERVE = Decimal('0.05')  # Conservative per-call reservation, NOT a guaranteed max charge.
MODEL = 'gpt-4.1-mini'
INPUT_RATE = Decimal('0.40') / Decimal('1000000')  # USD/token; verify before experiments.
OUTPUT_RATE = Decimal('1.60') / Decimal('1000000')
LEDGER = Path(__file__).resolve().parents[2] / 'data' / 'evaluation' / 'openai_usage.json'


def read_ledger():
    if not LEDGER.exists():
        return {'spent_usd': '0', 'entries': []}
    return json.loads(LEDGER.read_text(encoding='utf-8'))


def write_ledger(ledger):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix('.tmp')
    tmp.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    tmp.replace(LEDGER)


def reserve_call():
    ledger = read_ledger()
    spent = Decimal(ledger['spent_usd'])
    if spent + RESERVE > BUDGET:
        raise RuntimeError(f'Budget guard: ${spent} recorded; refusing a new ${RESERVE} reservation.')
    ledger['spent_usd'] = str(spent + RESERVE)
    ledger['entries'].append({'time': datetime.now(timezone.utc).isoformat(),
                              'model': MODEL, 'status': 'reserved', 'reserved_usd': str(RESERVE)})
    write_ledger(ledger)
    return len(ledger['entries']) - 1


def record_success(index, response):
    ledger = read_ledger()
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    estimated = Decimal(input_tokens) * INPUT_RATE + Decimal(output_tokens) * OUTPUT_RATE
    entry = ledger['entries'][index]
    entry.update({'status': 'completed', 'input_tokens': input_tokens,
                  'output_tokens': output_tokens, 'estimated_usd': str(estimated)})
    ledger['spent_usd'] = str(Decimal(ledger['spent_usd']) - RESERVE + max(estimated, Decimal('0')))
    write_ledger(ledger)
    return estimated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Explicitly allow one paid API call')
    args = parser.parse_args()
    ledger = read_ledger()
    spent = Decimal(ledger['spent_usd'])
    print(f'Model: {MODEL}; recorded experiment usage: ${spent}; budget: ${BUDGET}')
    if spent >= WARNING:
        print('WARNING: recorded usage has reached the $0.75 warning threshold.')
    if not args.live:
        print('OFFLINE MODE: No API request made. Run with --live for ONE paid smoke test.')
        return
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is not set. Do not paste it into source code.')
    # Reserve BEFORE calling API; failures keep the reservation for safety.
    index = reserve_call()
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=20.0)
    try:
        response = client.responses.create(
            model=MODEL,
            instructions='You are testing API connectivity. Reply only with the word READY.',
            input='Connectivity test. No SiteSeer or client data is included.',
            max_output_tokens=30,
            store=False,
        )
    except Exception:
        ledger = read_ledger()
        ledger['entries'][index]['status'] = 'uncertain_or_failed_keep_reservation'
        write_ledger(ledger)
        raise
    cost = record_success(index, response)
    print('Response:', response.output_text)
    print(f'Estimated cost: ${cost:.6f}; see local usage ledger.')
    if Decimal(read_ledger()['spent_usd']) >= WARNING:
        print('WARNING: recorded usage reached $0.75. Stop and review before more calls.')


if __name__ == '__main__':
    main()
