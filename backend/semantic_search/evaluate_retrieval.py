import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from search_pgvector import (
    get_database_connection,
    semantic_search_pgvector,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_test_set.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_test_data() -> list[dict]:
    """
    Load the synthetic retrieval evaluation dataset.
    """

    with TEST_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:

    print("SiteSeer 2B - Retrieval Evaluation")
    print("----------------------------------")

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    test_cases = load_test_data()

    connection = get_database_connection()

    total_answerable = 0
    hit_at_1 = 0
    hit_at_3 = 0
    filtered_tests = 0

    try:

        for test in test_cases:

            test_id = test["test_id"]
            question = test["question"]
            expected_ko_id = test["expected_ko_id"]

            filters = test.get("filters", {})

            # Insufficient-corpus questions are reserved for
            # the later answerability/refusal milestone.
            if expected_ko_id is None:
                continue

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
                learner_level=filters.get(
                    "learner_level"
                ),
            )

            retrieved_ids = [
                result["ko_id"]
                for result in results
            ]

            rank = None

            if expected_ko_id in retrieved_ids:
                rank = (
                    retrieved_ids.index(
                        expected_ko_id
                    )
                    + 1
                )

            if rank == 1:
                hit_at_1 += 1

            if rank is not None and rank <= 3:
                hit_at_3 += 1

            print()
            print(
                f"{test_id}: "
                f"{question}"
            )

            if filters:
                print(
                    f"Filters: {filters}"
                )
            else:
                print("Filters: None")

            print(
                f"Expected: {expected_ko_id}"
            )

            print(
                f"Retrieved: {retrieved_ids}"
            )

            if rank is None:
                print("Result: MISS")
            else:
                print(
                    f"Result: HIT at rank {rank}"
                )

    finally:
        connection.close()

    print()
    print("Evaluation Summary")
    print("------------------")

    print(
        f"Answerable questions: "
        f"{total_answerable}"
    )

    print(
        f"Metadata-filtered tests: "
        f"{filtered_tests}"
    )

    print(
        f"Hit@1: "
        f"{hit_at_1}/{total_answerable} "
        f"({hit_at_1 / total_answerable * 100:.1f}%)"
    )

    print(
        f"Hit@3: "
        f"{hit_at_3}/{total_answerable} "
        f"({hit_at_3 / total_answerable * 100:.1f}%)"
    )

    insufficient_count = sum(
        1
        for test in test_cases
        if test["expected_ko_id"] is None
    )

    print(
        f"Insufficient-corpus questions "
        f"reserved for later: "
        f"{insufficient_count}"
    )


if __name__ == "__main__":
    main()