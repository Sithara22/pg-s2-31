import json
from pathlib import Path
from typing import Any

from filter_indexable import filter_indexable_ekos
from loader import load_ekos
from validator import validate_ekos


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIRECTORY / "search_ready_ekos.json"


def clean_text(value: str) -> str:
    """
    Clean a text value before it is used for semantic retrieval.

    Current preprocessing is intentionally conservative:
    - removes leading/trailing whitespace
    - collapses repeated whitespace

    We do not lowercase or aggressively modify the knowledge text because
    the original learner-facing wording should remain available for RAG.
    """

    return " ".join(value.strip().split())


def build_searchable_text(record: dict[str, Any]) -> str:
    """
    Build the initial whole-object searchable text for one EKO.

    The baseline combines:
    - title
    - L1
    - L2
    - L3
    - L4
    - L5

    Labels are retained so the semantic representation preserves some
    information about the role of each knowledge level.
    """

    sections = [
        ("Title", record["title"]),
        ("L1 Factual Knowledge", record["l1"]),
        ("L2 Procedural Knowledge", record["l2"]),
        ("L3 Conditional Knowledge", record["l3"]),
        ("L4 Expert Judgement", record["l4"]),
        ("L5 Context and Dependencies", record["l5"]),
    ]

    searchable_sections = []

    for label, value in sections:
        cleaned_value = clean_text(value)

        if cleaned_value:
            searchable_sections.append(
                f"{label}: {cleaned_value}"
            )

    return "\n".join(searchable_sections)


def preprocess_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one approved EKO into the temporary search-ready format.
    """

    return {
        "ko_id": record["ko_id"],
        "searchable_text": build_searchable_text(record),
        "metadata": {
            "title": clean_text(record["title"]),
            "stage": clean_text(record["stage"]),
            "trade": clean_text(record["trade"]),
            "jurisdiction": clean_text(record["jurisdiction"]),
            "learner_level": clean_text(record["learner_level"]),
            "version": record["version"],
        },
        "source_content": {
            "title": clean_text(record["title"]),
            "l1": clean_text(record["l1"]),
            "l2": clean_text(record["l2"]),
            "l3": clean_text(record["l3"]),
            "l4": clean_text(record["l4"]),
            "l5": clean_text(record["l5"]),
        },
    }


def preprocess_ekos(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert approved EKO records into search-ready records.
    """

    return [
        preprocess_record(record)
        for record in records
    ]


def save_processed_ekos(
    records: list[dict[str, Any]],
    output_file: Path = OUTPUT_FILE,
) -> None:
    """
    Save search-ready EKO records as formatted JSON.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """
    Run the complete Milestone 1 preprocessing pipeline.
    """

    print("SiteSeer 2B - EKO Preprocessing Pipeline")
    print("----------------------------------------")

    # Step 1: Load
    ekos = load_ekos()

    print(f"Records loaded: {len(ekos)}")

    # Step 2: Validate
    validation_errors = validate_ekos(ekos)

    if validation_errors:
        print(f"Validation errors: {len(validation_errors)}")

        for error in validation_errors:
            print(f"- {error}")

        print()
        print("Pipeline stopped because validation failed.")
        return

    print("Validation errors: 0")

    # Step 3: Apply publication/indexability rules
    indexable_ekos, rejected_ekos = filter_indexable_ekos(ekos)

    print(f"Indexable records: {len(indexable_ekos)}")
    print(f"Rejected records: {len(rejected_ekos)}")

    # Step 4: Preprocess approved records
    search_ready_ekos = preprocess_ekos(indexable_ekos)

    print(
        f"Search-ready records: {len(search_ready_ekos)}"
    )

    # Step 5: Save derived dataset
    save_processed_ekos(search_ready_ekos)

    print(f"Output: {OUTPUT_FILE}")

    # Display first processed record for verification
    if search_ready_ekos:
        first_record = search_ready_ekos[0]

        print()
        print("Example Processed Record")
        print("------------------------")
        print(f"KO ID: {first_record['ko_id']}")
        print(
            f"Title: "
            f"{first_record['metadata']['title']}"
        )
        print()
        print("Searchable text:")
        print(first_record["searchable_text"])


if __name__ == "__main__":
    main()