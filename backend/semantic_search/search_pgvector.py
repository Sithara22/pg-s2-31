import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Database connection
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
# Semantic search with metadata filtering
# ---------------------------------------------------------

def semantic_search_pgvector(
    query: str,
    model: SentenceTransformer,
    connection,
    top_k: int = 3,
    stage: str | None = None,
    trade: str | None = None,
    jurisdiction: str | None = None,
    learner_level: str | None = None,
    eligible_ko_ids: list[str] | None = None,
) -> list[dict]:
    """
    Perform semantic search with optional metadata filters.

    Metadata filters restrict candidate Knowledge Objects
    before pgvector ranks them by cosine similarity.

    Returns EKO metadata, similarity score, and searchable
    L1-L5 evidence for downstream RAG generation.
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # None preserves the local demo; an empty allowlist must never broaden search.
    if eligible_ko_ids is not None and not eligible_ko_ids:
        return []

    # Convert learner question into an embedding.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # -----------------------------------------------------
    # Build optional SQL metadata filters
    # -----------------------------------------------------

    conditions = []
    filter_values = []

    if eligible_ko_ids is not None:
        conditions.append("ko_id = ANY(%s::text[])")
        filter_values.append(eligible_ko_ids)

    if stage:
        conditions.append("stage = %s")
        filter_values.append(stage)

    if trade:
        conditions.append("trade = %s")
        filter_values.append(trade)

    if jurisdiction:
        conditions.append("jurisdiction = %s")
        filter_values.append(jurisdiction)

    if learner_level:
        conditions.append("learner_level = %s")
        filter_values.append(learner_level)

    where_clause = ""

    if conditions:
        where_clause = (
            "WHERE " + " AND ".join(conditions)
        )

    # -----------------------------------------------------
    # Vector search
    # -----------------------------------------------------

    sql = f"""
        SELECT
            ko_id,
            title,
            searchable_text,
            stage,
            trade,
            jurisdiction,
            learner_level,
            version,
            1 - (embedding <=> %s) AS similarity
        FROM knowledge_embeddings
        {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s;
    """

    parameters = [
        query_embedding,
        *filter_values,
        query_embedding,
        top_k,
    ]

    with connection.cursor() as cursor:
        cursor.execute(sql, parameters)
        rows = cursor.fetchall()

    results = []

    for row in rows:
        results.append(
            {
                "ko_id": row[0],
                "title": row[1],
                "searchable_text": row[2],
                "stage": row[3],
                "trade": row[4],
                "jurisdiction": row[5],
                "learner_level": row[6],
                "version": row[7],
                "score": float(row[8]),
            }
        )

    return results


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(
    query: str,
    results: list[dict],
    filters: dict,
) -> None:

    print()
    print("Query")
    print("-----")
    print(query)

    print()
    print("Metadata Filters")
    print("----------------")

    active_filters = {
        key: value
        for key, value in filters.items()
        if value
    }

    if active_filters:
        for key, value in active_filters.items():
            print(f"{key}: {value}")
    else:
        print("None")

    print()
    print("Top Semantic Search Results")
    print("---------------------------")

    if not results:
        print(
            "No Knowledge Objects matched "
            "the selected metadata filters."
        )
        return

    for rank, result in enumerate(results, start=1):
        print()
        print(
            f"{rank}. "
            f"{result['ko_id']} | "
            f"{result['title']} | "
            f"score={result['score']:.4f}"
        )

        print(
            f"   Stage: {result['stage']} | "
            f"Trade: {result['trade']} | "
            f"Jurisdiction: {result['jurisdiction']} | "
            f"Learner level: {result['learner_level']}"
        )


# ---------------------------------------------------------
# Optional input helper
# ---------------------------------------------------------

def optional_input(prompt: str) -> str | None:
    """
    Blank input means that the metadata filter is not used.
    """

    value = input(prompt).strip()

    return value if value else None


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print(
        "SiteSeer 2B - Semantic Search "
        "with Metadata Filtering"
    )

    print(
        "-----------------------------------------------"
    )

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    connection = get_database_connection()

    try:
        print("Connected to PostgreSQL.")

        query = input(
            "\nEnter a learner question: "
        ).strip()

        print()
        print(
            "Optional metadata filters "
            "(press Enter to skip)"
        )
        print("--------------------------------")

        stage = optional_input("Stage: ")
        trade = optional_input("Trade: ")
        jurisdiction = optional_input("Jurisdiction: ")
        learner_level = optional_input("Learner level: ")

        filters = {
            "stage": stage,
            "trade": trade,
            "jurisdiction": jurisdiction,
            "learner_level": learner_level,
        }

        results = semantic_search_pgvector(
            query=query,
            model=model,
            connection=connection,
            top_k=3,
            stage=stage,
            trade=trade,
            jurisdiction=jurisdiction,
            learner_level=learner_level,
        )

        display_results(
            query=query,
            results=results,
            filters=filters,
        )

    finally:
        connection.close()

        print()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
