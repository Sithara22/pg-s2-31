import json
from pathlib import Path
from typing import Any

from filter_indexable import filter_indexable_ekos
from validator import validate_ekos


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "mock" / "ekos.json"


def load_ekos(file_path: Path = DEFAULT_DATA_PATH) -> list[dict[str, Any]]:
    """
    Load EKO records from a JSON file.

    Args:
        file_path:
            Path to the JSON file containing EKO records.

    Returns:
        A list of EKO records represented as dictionaries.

    Raises:
        FileNotFoundError:
            If the supplied JSON file does not exist.

        ValueError:
            If the JSON is invalid or the top-level JSON structure
            is not a list.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"EKO data file was not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in EKO data file: {file_path}"
        ) from error

    if not isinstance(data, list):
        raise ValueError(
            "EKO data must use a JSON array as the top-level structure."
        )

    return data


def main() -> None:
    """
    Load, validate and filter the mock EKO dataset.
    """

    print("SiteSeer 2B - EKO Data Pipeline")
    print("-------------------------------")

    ekos = load_ekos()

    print(f"Source: {DEFAULT_DATA_PATH}")
    print(f"Records loaded: {len(ekos)}")

    validation_errors = validate_ekos(ekos)

    if validation_errors:
        print()
        print(f"Validation errors: {len(validation_errors)}")

        for error in validation_errors:
            print(f"- {error}")

        print()
        print("Pipeline stopped because validation failed.")
        return

    print("Validation errors: 0")

    indexable_ekos, rejected_ekos = filter_indexable_ekos(ekos)

    print()
    print("Indexability Summary")
    print("--------------------")
    print(f"Indexable records: {len(indexable_ekos)}")
    print(f"Rejected records: {len(rejected_ekos)}")

    if rejected_ekos:
        print()
        print("Rejected Records")
        print("----------------")

        for rejected in rejected_ekos:
            reasons = ", ".join(rejected["reasons"])

            print(
                f"{rejected['ko_id']} -> {reasons}"
            )


if __name__ == "__main__":
    main()