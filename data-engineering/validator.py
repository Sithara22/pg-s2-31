from typing import Any


REQUIRED_FIELDS = {
    "ko_id",
    "title",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "stage",
    "trade",
    "jurisdiction",
    "learner_level",
    "publication_status",
    "validation_status",
    "ai_indexable",
    "superseded",
    "version",
}

STRING_FIELDS = {
    "ko_id",
    "title",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "stage",
    "trade",
    "jurisdiction",
    "learner_level",
    "publication_status",
    "validation_status",
}

BOOLEAN_FIELDS = {
    "ai_indexable",
    "superseded",
}

ALLOWED_PUBLICATION_STATUSES = {
    "published",
    "draft",
}

ALLOWED_VALIDATION_STATUSES = {
    "validated",
    "unvalidated",
}


def validate_record(record: dict[str, Any], index: int) -> list[str]:
    """
    Validate one EKO record against the temporary Project 2B schema.

    Returns:
        A list of validation errors.
        An empty list means the record passed validation.
    """

    errors = []

    # Check required fields.
    missing_fields = REQUIRED_FIELDS - record.keys()

    if missing_fields:
        errors.append(
            f"Record {index}: missing required fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

        # Stop further validation because some expected fields are absent.
        return errors

    # Check string field types.
    for field in STRING_FIELDS:
        if not isinstance(record[field], str):
            errors.append(
                f"Record {index} ({record.get('ko_id', 'UNKNOWN')}): "
                f"'{field}' must be a string."
            )

    # Check boolean field types.
    for field in BOOLEAN_FIELDS:
        if not isinstance(record[field], bool):
            errors.append(
                f"Record {index} ({record.get('ko_id', 'UNKNOWN')}): "
                f"'{field}' must be true or false."
            )

    # Version must be a positive integer.
    version = record["version"]

    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
    ):
        errors.append(
            f"Record {index} ({record.get('ko_id', 'UNKNOWN')}): "
            "'version' must be a positive integer."
        )

    # Important identity/display fields cannot be blank.
    for field in ("ko_id", "title"):
        value = record[field]

        if isinstance(value, str) and not value.strip():
            errors.append(
                f"Record {index}: '{field}' cannot be empty."
            )

    # At least one L1-L5 field must contain knowledge text.
    knowledge_fields = [
        record["l1"],
        record["l2"],
        record["l3"],
        record["l4"],
        record["l5"],
    ]

    has_knowledge = any(
        isinstance(value, str) and value.strip()
        for value in knowledge_fields
    )

    if not has_knowledge:
        errors.append(
            f"Record {index} ({record.get('ko_id', 'UNKNOWN')}): "
            "at least one of L1-L5 must contain text."
        )

    # Check temporary status values.
    publication_status = record["publication_status"]

    if (
        isinstance(publication_status, str)
        and publication_status not in ALLOWED_PUBLICATION_STATUSES
    ):
        errors.append(
            f"Record {index} ({record.get('ko_id', 'UNKNOWN')}): "
            f"unsupported publication_status "
            f"'{publication_status}'."
        )

    validation_status = record["validation_status"]

    if (
        isinstance(validation_status, str)
        and validation_status not in ALLOWED_VALIDATION_STATUSES
    ):
        errors.append(
            f"Record {index} ({record.get('ko_id', 'UNKNOWN')}): "
            f"unsupported validation_status "
            f"'{validation_status}'."
        )

    return errors


def validate_ekos(records: list[dict[str, Any]]) -> list[str]:
    """
    Validate all EKO records.

    Also checks that KO IDs are unique.
    """

    errors = []
    seen_ids = set()

    for index, record in enumerate(records, start=1):

        if not isinstance(record, dict):
            errors.append(
                f"Record {index}: expected a JSON object."
            )
            continue

        record_errors = validate_record(record, index)
        errors.extend(record_errors)

        ko_id = record.get("ko_id")

        if isinstance(ko_id, str) and ko_id.strip():

            if ko_id in seen_ids:
                errors.append(
                    f"Record {index}: duplicate ko_id '{ko_id}'."
                )
            else:
                seen_ids.add(ko_id)

    return errors