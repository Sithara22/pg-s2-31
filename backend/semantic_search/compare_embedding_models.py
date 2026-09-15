import json
import time
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

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_test_set.json"
)


# MiniLM is our current development baseline.
# MPNet is the challenger model.
MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
}


# ---------------------------------------------------------
# Load JSON
# ---------------------------------------------------------

def load_json(path: Path):
    """
    Load and return JSON data from a file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Evaluate one embedding model
# ---------------------------------------------------------

def evaluate_model(
    model_label: str,
    model_name: str,
    records: list[dict],
    test_questions: list[dict],
) -> dict:
    """
    Evaluate one embedding model using the same corpus
    and labelled retrieval questions.

    Metrics:
    - Hit@1
    - Hit@3
    - Mean Reciprocal Rank (MRR)
    - Model loading time
    - Corpus embedding time
    - Average query time
    """

    print()
    print("=" * 60)
    print(f"Evaluating: {model_label}")
    print(f"Model: {model_name}")
    print("=" * 60)

    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    start_time = time.perf_counter()

    model = SentenceTransformer(model_name)

    model_load_time = time.perf_counter() - start_time

    # -----------------------------------------------------
    # Prepare corpus
    # -----------------------------------------------------

    texts = [
        record["searchable_text"]
        for record in records
    ]

    ko_ids = [
        record["ko_id"]
        for record in records
    ]

    if not texts:
        raise ValueError(
            "No search-ready EKO records were found."
        )

    # -----------------------------------------------------
    # Generate corpus embeddings
    # -----------------------------------------------------

    start_time = time.perf_counter()

    corpus_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    corpus_embedding_time = (
        time.perf_counter() - start_time
    )

    print(
        f"Corpus embedding shape: "
        f"{corpus_embeddings.shape}"
    )

    # -----------------------------------------------------
    # Select answerable evaluation questions
    # -----------------------------------------------------

    answerable_questions = [
        question
        for question in test_questions
        if question.get("expected_answerability")
        == "answerable"
    ]

    if not answerable_questions:
        raise ValueError(
            "No answerable evaluation questions were found."
        )

    # -----------------------------------------------------
    # Evaluation counters
    # -----------------------------------------------------

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_sum = 0.0
    query_times = []

    # -----------------------------------------------------
    # Evaluate each question
    # -----------------------------------------------------

    for question_number, question in enumerate(
        answerable_questions,
        start=1,
    ):

        query_text = question.get("question")
        expected_ko = question.get("expected_ko_id")

        if not query_text:
            raise ValueError(
                "An evaluation record is missing 'question'."
            )

        if not expected_ko:
            raise ValueError(
                "An answerable evaluation record is missing "
                "'expected_ko_id'."
            )

        if expected_ko not in ko_ids:
            raise ValueError(
                f"Expected KO '{expected_ko}' is not present "
                "in the search-ready corpus."
            )

        # -------------------------------------------------
        # Generate query embedding
        # -------------------------------------------------

        query_start_time = time.perf_counter()

        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Corpus and query embeddings are normalized.
        # Therefore dot product gives cosine similarity.
        scores = corpus_embeddings @ query_embedding

        ranked_indices = np.argsort(scores)[::-1]

        query_elapsed_time = (
            time.perf_counter() - query_start_time
        )

        query_times.append(query_elapsed_time)

        # -------------------------------------------------
        # Convert ranked indexes to KO IDs
        # -------------------------------------------------

        ranked_ko_ids = [
            ko_ids[index]
            for index in ranked_indices
        ]

        expected_rank = (
            ranked_ko_ids.index(expected_ko) + 1
        )

        # -------------------------------------------------
        # Calculate retrieval metrics
        # -------------------------------------------------

        if expected_rank == 1:
            hit_at_1 += 1

        if expected_rank <= 3:
            hit_at_3 += 1

        reciprocal_rank_sum += 1 / expected_rank

        top_index = ranked_indices[0]
        top_result = ranked_ko_ids[0]
        top_score = float(scores[top_index])

        # -------------------------------------------------
        # Display question result
        # -------------------------------------------------

        # The current evaluation JSON may use "test_id",
        # "id", or neither. The identifier is only used
        # for display and does not affect evaluation.
        question_id = question.get(
            "test_id",
            question.get(
                "id",
                f"TEST-{question_number:03d}",
            ),
        )

        print()
        print(f"{question_id} | {query_text}")
        print(f"Expected KO: {expected_ko}")
        print(f"Expected KO rank: {expected_rank}")
        print(
            f"Top result: {top_result} "
            f"| score={top_score:.4f}"
        )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    total = len(answerable_questions)

    hit_at_1_percentage = (
        hit_at_1 / total * 100
    )

    hit_at_3_percentage = (
        hit_at_3 / total * 100
    )

    mrr = (
        reciprocal_rank_sum / total
    )

    average_query_time = (
        sum(query_times) / len(query_times)
    )

    return {
        "label": model_label,
        "model": model_name,
        "dimensions": int(
            corpus_embeddings.shape[1]
        ),
        "model_load_seconds": model_load_time,
        "corpus_embedding_seconds": (
            corpus_embedding_time
        ),
        "average_query_seconds": (
            average_query_time
        ),
        "hit_at_1": hit_at_1,
        "hit_at_1_percentage": (
            hit_at_1_percentage
        ),
        "hit_at_3": hit_at_3,
        "hit_at_3_percentage": (
            hit_at_3_percentage
        ),
        "mrr": mrr,
        "total": total,
    }


# ---------------------------------------------------------
# Display final comparison
# ---------------------------------------------------------

def display_summary(results: list[dict]) -> None:
    """
    Display a summary comparing all evaluated models.
    """

    print()
    print("=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)

    for result in results:

        print()
        print(result["label"])
        print("-" * 40)

        print(
            f"Model: {result['model']}"
        )

        print(
            f"Dimensions: {result['dimensions']}"
        )

        print(
            f"Hit@1: "
            f"{result['hit_at_1']}/"
            f"{result['total']} "
            f"("
            f"{result['hit_at_1_percentage']:.1f}%"
            f")"
        )

        print(
            f"Hit@3: "
            f"{result['hit_at_3']}/"
            f"{result['total']} "
            f"("
            f"{result['hit_at_3_percentage']:.1f}%"
            f")"
        )

        print(
            f"MRR: {result['mrr']:.4f}"
        )

        print(
            "Model load time: "
            f"{result['model_load_seconds']:.3f} seconds"
        )

        print(
            "Corpus embedding time: "
            f"{result['corpus_embedding_seconds']:.3f} seconds"
        )

        print(
            "Average query time: "
            f"{result['average_query_seconds']:.4f} seconds"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    """
    Compare candidate embedding models using the same
    SiteSeer development corpus and retrieval test set.
    """

    print(
        "SiteSeer 2B - Embedding Model Evaluation"
    )

    print(
        "----------------------------------------"
    )

    print(
        "NOTE: This evaluation uses synthetic development "
        "EKOs and questions."
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    records = load_json(DATA_FILE)

    test_questions = load_json(TEST_FILE)

    if not isinstance(records, list):
        raise ValueError(
            "Search-ready EKO data must be a JSON array."
        )

    if not isinstance(test_questions, list):
        raise ValueError(
            "Retrieval test data must be a JSON array."
        )

    print(
        f"Search-ready EKOs: {len(records)}"
    )

    print(
        f"Evaluation questions: "
        f"{len(test_questions)}"
    )

    # -----------------------------------------------------
    # Evaluate models
    # -----------------------------------------------------

    results = []

    for model_label, model_name in MODELS.items():

        result = evaluate_model(
            model_label=model_label,
            model_name=model_name,
            records=records,
            test_questions=test_questions,
        )

        results.append(result)

    # -----------------------------------------------------
    # Display comparison
    # -----------------------------------------------------

    display_summary(results)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These results are development evidence only."
    )

    print(
        "The selected embedding model must be "
        "re-evaluated when real published SiteSeer EKOs "
        "and representative learner questions become "
        "available."
    )


if __name__ == "__main__":
    main()