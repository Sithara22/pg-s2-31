import json
from pathlib import Path

import numpy as np
from backend.embeddings.embedding_client import RemoteEmbeddingModel

# ---------------------------------------------------------
# Paths and model configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "search_ready_ekos.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

OUTPUT_FILE = OUTPUT_DIR / "minilm_embeddings.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load search-ready EKOs
# ---------------------------------------------------------


def load_search_ready_ekos() -> list[dict]:
    """
    Load the search-ready EKO records produced by the
    Milestone 1 preprocessing pipeline.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Search-ready EKO file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Search-ready EKO data must be a JSON array.")

    if not records:
        raise ValueError("Search-ready EKO dataset is empty.")

    return records


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------


def generate_embeddings(
    records: list[dict],
) -> np.ndarray:
    """
    Convert each EKO's searchable text into a numerical
    embedding vector.
    """

    print(f"Loading embedding model: {MODEL_NAME}")

    model = RemoteEmbeddingModel(MODEL_NAME)

    texts = []

    for record in records:
        searchable_text = record.get("searchable_text", "").strip()

        if not searchable_text:
            raise ValueError(
                f"{record.get('ko_id', 'Unknown KO')} " "has no searchable_text."
            )

        texts.append(searchable_text)

    print(f"Generating embeddings for {len(texts)} EKOs...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------


def save_embeddings(
    records: list[dict],
    embeddings: np.ndarray,
) -> None:
    """
    Save the embedding matrix together with the KO IDs.

    Each row in the embedding matrix corresponds to the KO ID
    at the same position in the ko_ids array.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ko_ids = np.array([record["ko_id"] for record in records])

    np.savez_compressed(
        OUTPUT_FILE,
        ko_ids=ko_ids,
        embeddings=embeddings,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:
    print("SiteSeer 2B - Embedding Generation")
    print("----------------------------------")

    records = load_search_ready_ekos()

    print(f"Search-ready EKOs loaded: {len(records)}")

    embeddings = generate_embeddings(records)

    print()
    print("Embedding Summary")
    print("-----------------")
    print(f"Number of embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Array shape: {embeddings.shape}")

    save_embeddings(
        records,
        embeddings,
    )

    print()
    print(f"Embeddings saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
