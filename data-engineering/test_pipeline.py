from filter_indexable import filter_indexable_ekos
from loader import load_ekos
from preprocess import preprocess_ekos
from validator import validate_ekos


def run_tests() -> None:
    print("SiteSeer 2B - Milestone 1 Pipeline Tests")
    print("----------------------------------------")

    # ---------------------------------------------------------
    # Test 1: Loader
    # ---------------------------------------------------------

    records = load_ekos()

    assert len(records) == 15, (
        f"Expected 15 input records, got {len(records)}."
    )

    print("PASS: Loader returned 15 records.")

    # ---------------------------------------------------------
    # Test 2: Schema validation
    # ---------------------------------------------------------

    validation_errors = validate_ekos(records)

    assert len(validation_errors) == 0, (
        "Expected no schema validation errors, "
        f"got {len(validation_errors)}: {validation_errors}"
    )

    print("PASS: All 15 records passed schema validation.")

    # ---------------------------------------------------------
    # Test 3: Indexability filtering
    # ---------------------------------------------------------

    indexable_records, rejected_records = (
        filter_indexable_ekos(records)
    )

    assert len(indexable_records) == 10, (
        f"Expected 10 indexable records, "
        f"got {len(indexable_records)}."
    )

    assert len(rejected_records) == 5, (
        f"Expected 5 rejected records, "
        f"got {len(rejected_records)}."
    )

    print("PASS: 10 records accepted and 5 records rejected.")

    # ---------------------------------------------------------
    # Test 4: Correct records were rejected
    # ---------------------------------------------------------

    rejected_ids = {
        record["ko_id"]
        for record in rejected_records
    }

    expected_rejected_ids = {
        "MOCK-KO-011",
        "MOCK-KO-012",
        "MOCK-KO-013",
        "MOCK-KO-014",
        "MOCK-KO-015",
    }

    assert rejected_ids == expected_rejected_ids, (
        "Rejected KO IDs do not match expected values. "
        f"Expected {expected_rejected_ids}, got {rejected_ids}."
    )

    print("PASS: Correct non-indexable records were rejected.")

    # ---------------------------------------------------------
    # Test 5: Draft records cannot enter the index
    # ---------------------------------------------------------

    assert all(
        record["publication_status"] == "published"
        for record in indexable_records
    )

    print("PASS: No draft records entered the indexable dataset.")

    # ---------------------------------------------------------
    # Test 6: Only validated records enter the index
    # ---------------------------------------------------------

    assert all(
        record["validation_status"] == "validated"
        for record in indexable_records
    )

    print("PASS: All indexable records are validated.")

    # ---------------------------------------------------------
    # Test 7: Only AI-approved records enter the index
    # ---------------------------------------------------------

    assert all(
        record["ai_indexable"] is True
        for record in indexable_records
    )

    print("PASS: All indexable records are AI-indexable.")

    # ---------------------------------------------------------
    # Test 8: Superseded records cannot enter the index
    # ---------------------------------------------------------

    assert all(
        record["superseded"] is False
        for record in indexable_records
    )

    print("PASS: No superseded records entered the index.")

    # ---------------------------------------------------------
    # Test 9: Preprocessing
    # ---------------------------------------------------------

    search_ready_records = preprocess_ekos(
        indexable_records
    )

    assert len(search_ready_records) == 10, (
        f"Expected 10 search-ready records, "
        f"got {len(search_ready_records)}."
    )

    print("PASS: Preprocessing produced 10 search-ready records.")

    # ---------------------------------------------------------
    # Test 10: Searchable text exists
    # ---------------------------------------------------------

    for record in search_ready_records:

        assert record["searchable_text"].strip(), (
            f"{record['ko_id']} has empty searchable text."
        )

        assert "Title:" in record["searchable_text"], (
            f"{record['ko_id']} searchable text "
            "does not contain the title."
        )

    print("PASS: Every search-ready record contains searchable text.")

    # ---------------------------------------------------------
    # Test 11: Metadata is retained
    # ---------------------------------------------------------

    required_metadata = {
        "title",
        "stage",
        "trade",
        "jurisdiction",
        "learner_level",
        "version",
    }

    for record in search_ready_records:

        metadata_fields = set(
            record["metadata"].keys()
        )

        assert required_metadata.issubset(metadata_fields), (
            f"{record['ko_id']} is missing required metadata."
        )

    print("PASS: Required metadata is retained.")

    # ---------------------------------------------------------
    # Test 12: L1-L5 source content is retained
    # ---------------------------------------------------------

    required_source_fields = {
        "title",
        "l1",
        "l2",
        "l3",
        "l4",
        "l5",
    }

    for record in search_ready_records:

        source_fields = set(
            record["source_content"].keys()
        )

        assert required_source_fields.issubset(source_fields), (
            f"{record['ko_id']} is missing source content."
        )

    print("PASS: Original L1-L5 source content is retained.")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("All Milestone 1 pipeline tests passed.")


if __name__ == "__main__":
    run_tests()