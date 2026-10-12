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
    with TEST_DATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:

    print("SiteSeer 2B - Answerability Analysis")
    print("------------------------------------")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    test_cases = load_test_data()

    connection = get_database_connection()

    answerable_scores = []
    insufficient_scores = []

    try:

        for test in test_cases:

            question = test["question"]
            expected_ko_id = test["expected_ko_id"]
            filters = test.get("filters", {})

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

            if results:
                top_ko = results[0]["ko_id"]
                top_score = results[0]["score"]
            else:
                top_ko = None
                top_score = None

            if expected_ko_id is None:

                category = "INSUFFICIENT"

                if top_score is not None:
                    insufficient_scores.append(
                        top_score
                    )

            else:

                category = "ANSWERABLE"

                if top_score is not None:
                    answerable_scores.append(
                        top_score
                    )

            print()
            print(
                f"{test['test_id']} | {category}"
            )

            print(
                f"Question: {question}"
            )

            print(
                f"Top KO: {top_ko}"
            )

            if top_score is None:
                print("Top score: No result")
            else:
                print(
                    f"Top score: {top_score:.4f}"
                )

    finally:
        connection.close()

    print()
    print("Score Summary")
    print("-------------")

    if answerable_scores:

        print("Answerable:")

        print(
            f"  Minimum: "
            f"{min(answerable_scores):.4f}"
        )

        print(
            f"  Maximum: "
            f"{max(answerable_scores):.4f}"
        )

        print(
            f"  Average: "
            f"{sum(answerable_scores) / len(answerable_scores):.4f}"
        )

    if insufficient_scores:

        print()
        print("Insufficient corpus:")

        print(
            f"  Minimum: "
            f"{min(insufficient_scores):.4f}"
        )

        print(
            f"  Maximum: "
            f"{max(insufficient_scores):.4f}"
        )

        print(
            f"  Average: "
            f"{sum(insufficient_scores) / len(insufficient_scores):.4f}"
        )

    print()
    print(
        "These scores use synthetic development data "
        "and are not production thresholds."
    )


if __name__ == "__main__":
    main()