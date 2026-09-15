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
# pgvector semantic search
# ---------------------------------------------------------

def semantic_search_pgvector(
    query: str,
    model: SentenceTransformer,
    connection,
    top_k: int = 3,
) -> list[dict]:
    """
    Search the pgvector knowledge index using cosine distance.
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    # Convert the learner question into a vector using
    # the SAME embedding model used for the EKO corpus.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                ko_id,
                title,
                stage,
                trade,
                jurisdiction,
                learner_level,
                version,
                1 - (embedding <=> %s) AS similarity
            FROM knowledge_embeddings
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (
                query_embedding,
                query_embedding,
                top_k,
            ),
        )

        rows = cursor.fetchall()

    results = []

    for row in rows:
        results.append(
            {
                "ko_id": row[0],
                "title": row[1],
                "stage": row[2],
                "trade": row[3],
                "jurisdiction": row[4],
                "learner_level": row[5],
                "version": row[6],
                "score": float(row[7]),
            }
        )

    return results


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(
    query: str,
    results: list[dict],
) -> None:

    print()
    print("Query")
    print("-----")
    print(query)

    print()
    print("Top pgvector Semantic Search Results")
    print("------------------------------------")

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"{result['ko_id']} | "
            f"{result['title']} | "
            f"score={result['score']:.4f}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("SiteSeer 2B - pgvector Semantic Search")
    print("--------------------------------------")

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    connection = get_database_connection()

    try:
        print("Connected to PostgreSQL.")

        query = input(
            "\nEnter a learner question: "
        )

        results = semantic_search_pgvector(
            query=query,
            model=model,
            connection=connection,
            top_k=3,
        )

        display_results(
            query=query,
            results=results,
        )

    finally:
        connection.close()

        print()
        print("Database connection closed.")


if __name__ == "__main__":
    main()