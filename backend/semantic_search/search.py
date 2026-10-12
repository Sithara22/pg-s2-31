import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


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

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load search-ready EKO data
# ---------------------------------------------------------

def load_ekos() -> list[dict]:
    """
    Load the processed EKO records.
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
# Load stored embeddings
# ---------------------------------------------------------

def load_embeddings():
    """
    Load KO IDs and their previously generated embeddings.
    """

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_FILE}"
        )

    data = np.load(EMBEDDINGS_FILE)

    ko_ids = data["ko_ids"]
    embeddings = data["embeddings"]

    return ko_ids, embeddings


# ---------------------------------------------------------
# Semantic search
# ---------------------------------------------------------

def semantic_search(
    query: str,
    records: list[dict],
    ko_ids: np.ndarray,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """
    Embed a learner query and return the most semantically
    similar Knowledge Objects.
    """

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    # Convert the learner's question into an embedding
    # using the SAME model used for the EKO corpus.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Because both the stored embeddings and query embedding
    # are normalized, the dot product is equivalent to
    # cosine similarity.
    similarity_scores = embeddings @ query_embedding

    # Sort from highest similarity to lowest similarity.
    ranked_indices = np.argsort(similarity_scores)[::-1]

    # Prevent top_k from exceeding the number of records.
    top_k = min(top_k, len(ranked_indices))

    records_by_id = {
        record["ko_id"]: record
        for record in records
    }

    results = []

    for index in ranked_indices[:top_k]:
        ko_id = str(ko_ids[index])

        record = records_by_id.get(ko_id)

        if record is None:
            continue

        results.append(
            {
                "ko_id": ko_id,
                "title": record["metadata"]["title"],
                "score": float(similarity_scores[index]),
                "metadata": record["metadata"],
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
    """
    Print semantic search results in a readable format.
    """

    print()
    print("Query")
    print("-----")
    print(query)

    print()
    print("Top Semantic Search Results")
    print("---------------------------")

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
    print("SiteSeer 2B - Semantic Search")
    print("-----------------------------")

    records = load_ekos()

    ko_ids, embeddings = load_embeddings()

    if len(ko_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of KO IDs does not match number of embeddings."
        )

    print(f"Knowledge Objects loaded: {len(records)}")
    print(f"Embedding matrix: {embeddings.shape}")

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print()
    query = input("Enter a learner question: ")

    results = semantic_search(
        query=query,
        records=records,
        ko_ids=ko_ids,
        embeddings=embeddings,
        model=model,
        top_k=3,
    )

    display_results(
        query=query,
        results=results,
    )


if __name__ == "__main__":
    main()