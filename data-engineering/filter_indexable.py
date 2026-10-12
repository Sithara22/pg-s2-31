from typing import Any


def get_rejection_reasons(record: dict[str, Any]) -> list[str]:
    """
    Return the reasons why an EKO record should not enter the AI index.

    These rules are temporary Project 2B development assumptions.
    They must be reviewed when Project 2A provides the authoritative
    Knowledge Query Contract.
    """

    reasons = []

    if record["publication_status"] != "published":
        reasons.append(
            f"publication_status={record['publication_status']}"
        )

    if record["validation_status"] != "validated":
        reasons.append(
            f"validation_status={record['validation_status']}"
        )

    if record["ai_indexable"] is not True:
        reasons.append(
            f"ai_indexable={record['ai_indexable']}"
        )

    if record["superseded"] is True:
        reasons.append(
            f"superseded={record['superseded']}"
        )

    return reasons


def filter_indexable_ekos(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Separate EKO records into indexable and rejected groups.

    Returns:
        A tuple containing:

        1. indexable_records
        2. rejected_records

    Each rejected record contains its KO ID and rejection reasons.
    """

    indexable_records = []
    rejected_records = []

    for record in records:
        reasons = get_rejection_reasons(record)

        if reasons:
            rejected_records.append(
                {
                    "ko_id": record["ko_id"],
                    "reasons": reasons,
                }
            )
        else:
            indexable_records.append(record)

    return indexable_records, rejected_records