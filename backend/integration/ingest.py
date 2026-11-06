"""Validate a fixture or fetch a complete 2A snapshot; optionally refresh pgvector."""

import argparse
import json
from pathlib import Path

from backend.integration.team2a import Team2AClient, map_bulk_snapshot


def refresh_index(records):
    from backend.semantic_search.generate_embeddings import generate_embeddings
    from backend.semantic_search.index_pgvector import (
        create_table, get_database_connection, index_ekos, validate_index_inputs, verify_index,
    )
    import numpy as np

    embeddings = generate_embeddings(records)
    ids = np.array([record["ko_id"] for record in records])
    validate_index_inputs(records, ids, embeddings)
    connection = get_database_connection()
    try:
        create_table(connection)
        index_ekos(connection, records, ids, embeddings)
        verify_index(connection, set(ids.tolist()))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Use a provisional JSON snapshot instead of an API")
    parser.add_argument("--apply", action="store_true", help="Replace the entire derived index after validation")
    args = parser.parse_args()
    payload = json.loads(args.fixture.read_text(encoding="utf-8")) if args.fixture else Team2AClient().fetch_bulk()
    records = map_bulk_snapshot(payload)
    print(f"Validated {len(records)} eligible current EKO versions.")
    if args.apply:
        refresh_index(records)
    else:
        print("Validation only. Use --apply to refresh the derived index.")


if __name__ == "__main__":
    main()
