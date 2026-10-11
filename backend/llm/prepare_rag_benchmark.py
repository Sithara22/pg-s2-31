"""Prepare a fixed-evidence RAG benchmark using synthetic SiteSeer EKOs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EKO_PATH = ROOT / 'data/mock/ekos.json'
TEST_PATH = ROOT / 'data/evaluation/retrieval_test_set.json'
OUTPUT_PATH = ROOT / 'data/evaluation/rag_fixed_evidence_benchmark.json'
FIELDS = ('l1', 'l2', 'l3', 'l4', 'l5')

SYSTEM_INSTRUCTION = (
    'You are a SiteSeer learning assistant evaluating SYNTHETIC construction knowledge. '
    'Answer only using the EKO evidence provided. Do not rely on outside knowledge. '
    'If the evidence does not support a complete answer, say that the available EKO '
    'evidence is insufficient. Do not guess technical specifications, legal requirements, '
    'measurements, or standards. Cite supporting KO IDs in square brackets, e.g. '
    '[MOCK-KO-001]. Do not cite an EKO that does not support your claim.'
)


def read_array(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(f'Expected an array of objects: {path}')
    return data


def is_mock_eligible(eko: dict) -> bool:
    """Apply the provisional eligibility rules for synthetic mock records."""
    return (
        eko.get('publication_status') == 'published'
        and eko.get('validation_status') == 'validated'
        and eko.get('ai_indexable') is True
        and eko.get('superseded') is False
    )


def evidence_for(eko: dict) -> dict:
    return {
        'ko_id': eko['ko_id'],
        'title': eko['title'],
        'stage': eko.get('stage'),
        'trade': eko.get('trade'),
        'jurisdiction': eko.get('jurisdiction'),
        'version': eko.get('version'),
        'knowledge': {field.upper(): eko.get(field, '') for field in FIELDS},
    }


def render_prompt(question: str, evidence: list[dict]) -> str:
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2)
    return f'Question: {question}\n\nAvailable EKO evidence (synthetic):\n{rendered}'


def prepare(ekos: list[dict], tests: list[dict]) -> list[dict]:
    eligible = {e['ko_id']: e for e in ekos if is_mock_eligible(e)}
    seen = set()
    result = []
    for test in tests:
        test_id = test['test_id']
        if test_id in seen:
            raise ValueError(f'Duplicate test ID: {test_id}')
        seen.add(test_id)
        question = test['question']
        expected = test['expected_ko_id']
        filters = test.get('filters', {})
        if not isinstance(filters, dict):
            raise ValueError(f'Invalid filters in {test_id}')
        if expected is not None:
            if expected not in eligible:
                raise ValueError(f'{test_id}: expected EKO is not mock-eligible: {expected}')
            eko = eligible[expected]
            for key, value in filters.items():
                if eko.get(key) != value:
                    raise ValueError(f'{test_id}: expected EKO fails {key} filter')
            evidence = [evidence_for(eko)]
            label = 'answerable_candidate'
        else:
            evidence = []
            label = 'insufficient_zero_evidence'
        result.append({
            'test_id': test_id,
            'question': question,
            'filters': filters,
            'scenario': label,
            'expected_ko_id': expected,
            'model_input': {
                'system': SYSTEM_INSTRUCTION,
                'user': render_prompt(question, evidence),
            },
            'evidence_ko_ids': [e['ko_id'] for e in evidence],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ekos', type=Path, default=EKO_PATH)
    parser.add_argument('--tests', type=Path, default=TEST_PATH)
    parser.add_argument('--output', type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    records = prepare(read_array(args.ekos), read_array(args.tests))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    counts = {name: sum(x['scenario'] == name for x in records) for name in sorted({x['scenario'] for x in records})}
    print('SiteSeer 2B | RAG Benchmark Preparation')
    print('----------------------------------------')
    print(f'Total test cases:       {len(records)}')
    print(f"Answerable candidates:  {counts.get('answerable_candidate', 0)}")
    print(f"Insufficient evidence:   {counts.get('insufficient_zero_evidence', 0)}")
    try:
        output_label = args.output.relative_to(ROOT)
    except ValueError:
        output_label = args.output
    print(f'Output: {output_label}')
    print('Status: Completed')


if __name__ == '__main__':
    main()
