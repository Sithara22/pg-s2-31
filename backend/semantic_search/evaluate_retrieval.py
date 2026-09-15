import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from search import (
    MODEL_NAME,
    load_ekos,
    load_embeddings,
    semantic_search,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_SET_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_test_set.json"
)


# ---------------------------------------------------------
# Load evaluation dataset
# ---------------------------------------------------------

def load_test_set() -> list[dict]:
    """
    Load the temporary labelled retrieval evaluation dataset.
    """

    if not TEST_SET_FILE.exists():
        raise FileNotFoundError(
            f"Retrieval test set not found: {TEST_SET_FILE}"
        )

    with TEST_SET_FILE.open("r", encoding="utf-8") as file:
        tests = json.load(file)

    if not isinstance(tests, list):
        raise ValueError(
            "Retrieval test set must be a JSON array."
        )

    return tests


# ---------------------------------------------------------
# Evaluate answerable retrieval questions
# ---------------------------------------------------------

def evaluate_retrieval(
    tests: list[dict],
    records: list[dict],
    ko_ids: np.ndarray,
    embeddings: np.ndarray,
    model: SentenceTransformer,
) -> None:
    """
    Evaluate semantic retrieval using labelled answerable
    development questions.

    Hit@1:
        Expected KO is ranked first.

    Hit@3:
        Expected KO appears anywhere in the top three.

    Corpus-insufficient questions are reported separately.
    They are not used for Hit@1 / Hit@3 because retrieval
    thresholds and answerability logic have not been
    implemented yet.
    """

    answerable_tests = [
        test
        for test in tests
        if test["expected_answerability"] == "answerable"
    ]

    insufficient_tests = [
        test
        for test in tests
        if test["expected_answerability"] == "insufficient_corpus"
    ]

    hit_at_1 = 0
    hit_at_3 = 0

    print()
    print("Answerable Retrieval Tests")
    print("--------------------------")

    for test in answerable_tests:
        results = semantic_search(
            query=test["question"],
            records=records,
            ko_ids=ko_ids,
            embeddings=embeddings,
            model=model,
            top_k=3,
        )

        retrieved_ids = [
            result["ko_id"]
            for result in results
        ]

        expected_ko = test["expected_ko_id"]

        top_1_hit = (
            len(retrieved_ids) >= 1
            and retrieved_ids[0] == expected_ko
        )

        top_3_hit = expected_ko in retrieved_ids

        if top_1_hit:
            hit_at_1 += 1

        if top_3_hit:
            hit_at_3 += 1

        top_result = results[0] if results else None

        top_result_text = (
            f"{top_result['ko_id']} "
            f"({top_result['score']:.4f})"
            if top_result
            else "No result"
        )

        print()
        print(f"{test['test_id']}: {test['question']}")
        print(f"Expected: {expected_ko}")
        print(f"Top result: {top_result_text}")
        print(f"Hit@1: {'YES' if top_1_hit else 'NO'}")
        print(f"Hit@3: {'YES' if top_3_hit else 'NO'}")

        if not top_1_hit:
            print(
                "Top 3:",
                ", ".join(retrieved_ids)
                if retrieved_ids
                else "None",
            )

    total_answerable = len(answerable_tests)

    hit_at_1_rate = (
        hit_at_1 / total_answerable
        if total_answerable
        else 0
    )

    hit_at_3_rate = (
        hit_at_3 / total_answerable
        if total_answerable
        else 0
    )

    print()
    print("=" * 50)
    print("Development Retrieval Evaluation")
    print("=" * 50)

    print(f"Answerable test questions: {total_answerable}")

    print(
        f"Hit@1: {hit_at_1}/{total_answerable} "
        f"({hit_at_1_rate:.1%})"
    )

    print(
        f"Hit@3: {hit_at_3}/{total_answerable} "
        f"({hit_at_3_rate:.1%})"
    )

    print()
    print("Corpus-Insufficient Questions")
    print("-----------------------------")

    print(
        f"{len(insufficient_tests)} questions are reserved "
        "for later answerability/refusal evaluation."
    )

    print(
        "They are not scored yet because no evidence-based "
        "retrieval threshold or answerability classifier "
        "has been implemented."
    )

    print()
    print(
        "NOTE: These results use synthetic mock EKOs and "
        "development questions. They are not final SiteSeer "
        "retrieval-performance results."
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    print("SiteSeer 2B - Retrieval Evaluation")
    print("----------------------------------")

    tests = load_test_set()

    records = load_ekos()

    ko_ids, embeddings = load_embeddings()

    if len(ko_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of KO IDs does not match number of embeddings."
        )

    print(f"Evaluation questions loaded: {len(tests)}")
    print(f"Knowledge Objects loaded: {len(records)}")
    print(f"Embedding matrix: {embeddings.shape}")
    print(f"Embedding model: {MODEL_NAME}")

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    evaluate_retrieval(
        tests=tests,
        records=records,
        ko_ids=ko_ids,
        embeddings=embeddings,
        model=model,
    )


if __name__ == "__main__":
    main()