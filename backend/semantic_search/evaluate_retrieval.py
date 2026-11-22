
import json
from pathlib import Path

from backend.embeddings.embedding_client import RemoteEmbeddingModel

from search_pgvector import (
    get_database_connection,
    semantic_search_pgvector,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_test_set.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SUPPORTED_FILTERS = {
    "stage",
    "trade",
    "jurisdiction",
    "learner_level",
}


# ---------------------------------------------------------
# Load and validate evaluation data
# ---------------------------------------------------------

def load_test_data() -> list[dict]:
    """
    Load and validate synthetic retrieval test cases.
    """

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {TEST_DATA_PATH}"
        )

    with TEST_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError(
            "Evaluation dataset must be a non-empty JSON array."
        )

    seen_test_ids = set()

    for index, test in enumerate(test_cases, start=1):

        if not isinstance(test, dict):
            raise ValueError(
                f"Test case {index} must be an object."
            )

        required_fields = {
            "test_id",
            "question",
            "expected_ko_id",
            "filters",
        }

        missing_fields = required_fields - set(test)

        if missing_fields:
            raise ValueError(
                f"Test case {index} is missing: "
                f"{sorted(missing_fields)}"
            )

        test_id = test["test_id"]
        question = test["question"]
        expected_ko_id = test["expected_ko_id"]
        filters = test["filters"]

        if not isinstance(test_id, str) or not test_id.strip():
            raise ValueError(
                f"Invalid test_id in test case {index}."
            )

        if test_id in seen_test_ids:
            raise ValueError(
                f"Duplicate test ID: {test_id}"
            )

        seen_test_ids.add(test_id)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Invalid question in {test_id}."
            )

        if expected_ko_id is not None:
            if (
                not isinstance(expected_ko_id, str)
                or not expected_ko_id.strip()
            ):
                raise ValueError(
                    f"Invalid expected_ko_id in {test_id}."
                )

        if not isinstance(filters, dict):
            raise ValueError(
                f"Filters must be an object in {test_id}."
            )

        unknown_filters = set(filters) - SUPPORTED_FILTERS

        if unknown_filters:
            raise ValueError(
                f"Unsupported filters in {test_id}: "
                f"{sorted(unknown_filters)}"
            )

        for name, value in filters.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Invalid filter value for {name} in {test_id}."
                )

    return test_cases


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def main() -> None:

    print("SiteSeer 2B - Retrieval Evaluation")
    print("----------------------------------")

    test_cases = load_test_data()

    answerable_tests = [
        test
        for test in test_cases
        if test["expected_ko_id"] is not None
    ]

    insufficient_tests = [
        test
        for test in test_cases
        if test["expected_ko_id"] is None
    ]

    if not answerable_tests:
        raise ValueError(
            "No answerable test cases available for "
            "retrieval evaluation."
        )

    print(f"Total test cases loaded: {len(test_cases)}")
    print(f"Answerable cases: {len(answerable_tests)}")
    print(
        f"Insufficient-corpus cases reserved: "
        f"{len(insufficient_tests)}"
    )

    print(f"Loading model: {MODEL_NAME}")

    model = RemoteEmbeddingModel(MODEL_NAME)

    total_answerable = 0
    hit_at_1 = 0
    hit_at_3 = 0
    filtered_tests = 0

    missed_tests = []

    connection = get_database_connection()

    try:

        for test in answerable_tests:

            test_id = test["test_id"]
            question = test["question"]
            expected_ko_id = test["expected_ko_id"]
            filters = test["filters"]

            total_answerable += 1

            if filters:
                filtered_tests += 1

            results = semantic_search_pgvector(
                query=question,
                model=model,
                connection=connection,
                top_k=3,
                stage=filters.get("stage"),
                trade=filters.get("trade"),
                jurisdiction=filters.get("jurisdiction"),
                learner_level=filters.get("learner_level"),
            )

            retrieved_ids = [
                result["ko_id"]
                for result in results
            ]

            rank = None

            if expected_ko_id in retrieved_ids:
                rank = retrieved_ids.index(expected_ko_id) + 1

            if rank == 1:
                hit_at_1 += 1

            if rank is not None and rank <= 3:
                hit_at_3 += 1

            if rank != 1:
                missed_tests.append({
                    "test_id": test_id,
                    "question": question,
                    "expected": expected_ko_id,
                    "retrieved": retrieved_ids,
                    "rank": rank,
                })

            print()
            print(f"{test_id}: {question}")
            print(f"Filters: {filters if filters else 'None'}")
            print(f"Expected: {expected_ko_id}")
            print(f"Retrieved: {retrieved_ids}")

            if rank is None:
                print("Result: MISS - not in top 3")
            else:
                print(f"Result: HIT at rank {rank}")

    finally:
        connection.close()

    # -----------------------------------------------------
    # Evaluation summary
    # -----------------------------------------------------

    print()
    print("Evaluation Summary")
    print("------------------")

    print(f"Total test cases: {len(test_cases)}")
    print(f"Answerable questions: {total_answerable}")
    print(f"Metadata-filtered tests: {filtered_tests}")

    print(
        f"Hit@1: {hit_at_1}/{total_answerable} "
        f"({hit_at_1 / total_answerable * 100:.1f}%)"
    )

    print(
        f"Hit@3: {hit_at_3}/{total_answerable} "
        f"({hit_at_3 / total_answerable * 100:.1f}%)"
    )

    print(
        "Insufficient-corpus questions reserved for later: "
        f"{len(insufficient_tests)}"
    )

    print()
    print("Retrieval Errors")
    print("----------------")

    if not missed_tests:
        print("No top-1 retrieval errors.")
    else:
        print(
            f"Questions not retrieved at rank 1: "
            f"{len(missed_tests)}"
        )

        for miss in missed_tests:
            rank_label = (
                str(miss["rank"])
                if miss["rank"] is not None
                else "Not in top 3"
            )

            print()
            print(f"Test: {miss['test_id']}")
            print(f"Question: {miss['question']}")
            print(f"Expected: {miss['expected']}")
            print(f"Retrieved: {miss['retrieved']}")
            print(f"Rank: {rank_label}")


if __name__ == "__main__":
    main()