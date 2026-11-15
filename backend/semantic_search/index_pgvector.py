import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "search_ready_ekos.json"
)

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "minilm_embeddings.npz"
)

ENV_FILE = PROJECT_ROOT / ".env"

EMBEDDING_DIMENSION = 384


# ---------------------------------------------------------
# Environment / database connection
# ---------------------------------------------------------

def get_database_connection():
    """
    Connect to the local SiteSeer 2B PostgreSQL database.
    """

    load_dotenv(ENV_FILE)

    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    if not db_name or not db_user:
        raise ValueError(
            "DB_NAME and DB_USER must be configured in .env"
        )

    connection = psycopg.connect(
        dbname=db_name,
        user=db_user,
        host=db_host,
        port=db_port,
        password = os.getenv("DB_PASSWORD")
    )

    register_vector(connection)

    return connection


# ---------------------------------------------------------
# Load processed EKOs
# ---------------------------------------------------------

def load_ekos() -> list[dict]:
    """
    Load search-ready EKOs produced by preprocessing.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Search-ready EKO file not found: {DATA_FILE}"
        )

    with DATA_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Search-ready EKO data must be a JSON array."
        )

    return records


# ---------------------------------------------------------
# Load generated embeddings
# ---------------------------------------------------------

def load_embeddings():
    """
    Load MiniLM embeddings and their KO identifiers.
    """

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_FILE}"
        )

    with np.load(EMBEDDINGS_FILE, allow_pickle=False) as data:

        ko_ids = data["ko_ids"].copy()
        embeddings = data["embeddings"].copy()

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding data must be a two-dimensional matrix."
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, "
            f"found {embeddings.shape[1]}."
        )

    if len(ko_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of KO IDs does not match embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    return ko_ids, embeddings


# ---------------------------------------------------------
# Validate dataset and embedding alignment
# ---------------------------------------------------------

def validate_index_inputs(
    records: list[dict],
    ko_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    """
    Confirm that every search-ready EKO has exactly
    one corresponding embedding before modifying the DB.
    """

    if not records:
        raise ValueError(
            "Search-ready dataset is empty. "
            "Refusing to refresh the index."
        )

    record_ids = [record["ko_id"] for record in records]

    embedding_ids = [str(ko_id) for ko_id in ko_ids]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "Duplicate KO IDs found in search-ready data."
        )

    if len(embedding_ids) != len(set(embedding_ids)):
        raise ValueError(
            "Duplicate KO IDs found in embedding data."
        )

    record_id_set = set(record_ids)
    embedding_id_set = set(embedding_ids)

    if record_id_set != embedding_id_set:

        missing_embeddings = sorted(
            record_id_set - embedding_id_set
        )

        unexpected_embeddings = sorted(
            embedding_id_set - record_id_set
        )

        raise ValueError(
            "Search-ready EKOs and embeddings do not match. "
            f"Missing embeddings: {missing_embeddings}. "
            f"Unexpected embeddings: {unexpected_embeddings}."
        )

    if embeddings.shape[0] != len(records):
        raise ValueError(
            "Embedding count does not match EKO count."
        )

    for record in records:

        if not record["searchable_text"].strip():
            raise ValueError(
                f"Empty searchable text for {record['ko_id']}."
            )

        metadata = record["metadata"]

        if not metadata["title"].strip():
            raise ValueError(
                f"Missing title for {record['ko_id']}."
            )

    print("PASS: Dataset and embedding alignment verified.")


# ---------------------------------------------------------
# Create vector table
# ---------------------------------------------------------

def create_table(connection) -> None:
    """
    Create the derived semantic-search index.

    This table is not the authoritative SiteSeer
    knowledge source.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                ko_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                stage TEXT,
                trade TEXT,
                jurisdiction TEXT,
                learner_level TEXT,
                version INTEGER,
                embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL
            );
            """
        )


# ---------------------------------------------------------
# Refresh pgvector index
# ---------------------------------------------------------

def index_ekos(
    connection,
    records: list[dict],
    ko_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    """
    Synchronise the derived index with the complete
    search-ready dataset.

    Insert new KOs, update existing KOs, and remove
    records absent from the current source snapshot.

    The caller commits only after successful verification.
    """

    records_by_id = {
        record["ko_id"]: record
        for record in records
    }

    current_ids = [
        str(ko_id)
        for ko_id in ko_ids
    ]

    with connection.cursor() as cursor:

        for index, ko_id in enumerate(current_ids):

            record = records_by_id[ko_id]
            metadata = record["metadata"]
            embedding = embeddings[index]

            cursor.execute(
                """
                INSERT INTO knowledge_embeddings (
                    ko_id,
                    title,
                    searchable_text,
                    stage,
                    trade,
                    jurisdiction,
                    learner_level,
                    version,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (ko_id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    searchable_text = EXCLUDED.searchable_text,
                    stage = EXCLUDED.stage,
                    trade = EXCLUDED.trade,
                    jurisdiction = EXCLUDED.jurisdiction,
                    learner_level = EXCLUDED.learner_level,
                    version = EXCLUDED.version,
                    embedding = EXCLUDED.embedding;
                """,
                (
                    ko_id,
                    metadata["title"],
                    record["searchable_text"],
                    metadata.get("stage"),
                    metadata.get("trade"),
                    metadata.get("jurisdiction"),
                    metadata.get("learner_level"),
                    metadata.get("version"),
                    embedding,
                ),
            )

        # Remove KOs that are no longer in the
        # current approved search-ready snapshot.
        cursor.execute(
            """
            DELETE FROM knowledge_embeddings
            WHERE NOT (ko_id = ANY(%s::text[]));
            """,
            (current_ids,),
        )

        removed_count = cursor.rowcount

    print(f"EKOs inserted/updated: {len(current_ids)}")
    print(f"Stale EKOs removed: {removed_count}")


# ---------------------------------------------------------
# Verify index
# ---------------------------------------------------------

def verify_index(
    connection,
    expected_ids: set[str],
) -> None:
    """
    Verify that the database contains exactly the
    expected KO IDs with valid embedding dimensions.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                ko_id,
                title,
                vector_dims(embedding)
            FROM knowledge_embeddings
            ORDER BY ko_id;
            """
        )

        rows = cursor.fetchall()

    actual_ids = {
        row[0]
        for row in rows
    }

    if actual_ids != expected_ids:
        raise ValueError(
            "PostgreSQL KO IDs do not match the "
            "current search-ready dataset."
        )

    for ko_id, title, dimensions in rows:

        if dimensions != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Invalid embedding dimensions for {ko_id}: "
                f"{dimensions}"
            )

    print()
    print("pgvector Index Summary")
    print("----------------------")
    print(f"Total indexed EKOs: {len(rows)}")
    print("PASS: All KO IDs match the source dataset.")
    print("PASS: All embedding dimensions are valid.")

    print()
    print("Sample indexed records:")

    for ko_id, title, dimensions in rows[:5]:

        print(
            f"{ko_id} | {title} | "
            f"dimensions={dimensions}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("SiteSeer 2B - pgvector Indexing")
    print("-------------------------------")

    records = load_ekos()
    ko_ids, embeddings = load_embeddings()

    print(f"Search-ready EKOs loaded: {len(records)}")
    print(f"Embeddings loaded: {embeddings.shape}")

    # Validate before making any database changes.
    validate_index_inputs(
        records=records,
        ko_ids=ko_ids,
        embeddings=embeddings,
    )

    connection = get_database_connection()

    try:
        print("Connected to PostgreSQL.")

        # Create, refresh and verify within one transaction.
        create_table(connection)

        print("knowledge_embeddings table ready.")

        index_ekos(
            connection=connection,
            records=records,
            ko_ids=ko_ids,
            embeddings=embeddings,
        )

        verify_index(
            connection=connection,
            expected_ids={
                record["ko_id"]
                for record in records
            },
        )

        connection.commit()

        print()
        print("PASS: Index refresh committed successfully.")

    except Exception:

        connection.rollback()

        print()
        print("ERROR: Index refresh failed. Transaction rolled back.")

        raise

    finally:

        connection.close()

        print("Database connection closed.")


if __name__ == "__main__":
    main()
