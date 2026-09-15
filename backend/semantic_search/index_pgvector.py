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
    )

    register_vector(connection)

    return connection


# ---------------------------------------------------------
# Load processed EKOs
# ---------------------------------------------------------

def load_ekos() -> list[dict]:
    """
    Load search-ready EKOs produced by the preprocessing
    pipeline.
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
    Load the MiniLM embeddings and corresponding KO IDs.
    """

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_FILE}"
        )

    data = np.load(EMBEDDINGS_FILE)

    ko_ids = data["ko_ids"]
    embeddings = data["embeddings"]

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding data must be a two-dimensional matrix."
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings, "
            f"but found {embeddings.shape[1]} dimensions."
        )

    if len(ko_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of KO IDs does not match number of embeddings."
        )

    return ko_ids, embeddings


# ---------------------------------------------------------
# Create vector table
# ---------------------------------------------------------

def create_table(connection) -> None:
    """
    Create the derived SiteSeer semantic-search index.

    This table is not the authoritative knowledge source.
    It can be rebuilt from approved EKO data.
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

    connection.commit()


# ---------------------------------------------------------
# Index EKOs
# ---------------------------------------------------------

def index_ekos(
    connection,
    records: list[dict],
    ko_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    """
    Insert or update each search-ready EKO and its embedding.

    ON CONFLICT allows the derived index to be refreshed when
    the same KO is indexed again.
    """

    records_by_id = {
        record["ko_id"]: record
        for record in records
    }

    indexed_count = 0

    with connection.cursor() as cursor:

        for index, ko_id_value in enumerate(ko_ids):

            ko_id = str(ko_id_value)

            record = records_by_id.get(ko_id)

            if record is None:
                raise ValueError(
                    f"No search-ready EKO found for {ko_id}"
                )

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

            indexed_count += 1

    connection.commit()

    print(f"EKOs indexed into pgvector: {indexed_count}")


# ---------------------------------------------------------
# Verify index
# ---------------------------------------------------------

def verify_index(connection) -> None:
    """
    Display basic information about the pgvector index.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM knowledge_embeddings;
            """
        )

        total = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT
                ko_id,
                title,
                vector_dims(embedding)
            FROM knowledge_embeddings
            ORDER BY ko_id
            LIMIT 5;
            """
        )

        sample_records = cursor.fetchall()

    print()
    print("pgvector Index Summary")
    print("----------------------")
    print(f"Total indexed EKOs: {total}")

    print()
    print("Sample indexed records:")

    for ko_id, title, dimensions in sample_records:
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

    connection = get_database_connection()

    try:
        print("Connected to PostgreSQL.")

        create_table(connection)

        print("knowledge_embeddings table ready.")

        index_ekos(
            connection=connection,
            records=records,
            ko_ids=ko_ids,
            embeddings=embeddings,
        )

        verify_index(connection)

    finally:
        connection.close()

        print()
        print("Database connection closed.")


if __name__ == "__main__":
    main()