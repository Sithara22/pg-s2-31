
import os

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class RemoteEmbeddingModel:
    """
    Remote replacement for SentenceTransformer.

    Supports the .encode() arguments used by SiteSeer,
    without loading any model weights locally.
    """

    def __init__(self, model_name=MODEL_NAME):
        if model_name != MODEL_NAME:
            raise ValueError(
                f"Only {MODEL_NAME} is configured "
                "for remote inference."
            )

        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN is missing from .env"
            )

        self.model_name = model_name

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=token,
            timeout=120,
        )

    def encode(
        self,
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16,
    ):
        """
        Generate embeddings remotely.

        String input  -> NumPy vector (384,)
        List input    -> NumPy matrix (N, 384)
        """

        if not convert_to_numpy:
            raise ValueError(
                "SiteSeer expects NumPy embeddings."
            )

        single_input = isinstance(sentences, str)

        if single_input:
            texts = [sentences]
        else:
            texts = list(sentences)

        if not texts:
            return np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32
            )

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All embedding inputs must be "
                "non-empty strings."
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            response = self.client.feature_extraction(
                text=batch,
                model=self.model_name,
            )

            vectors = np.asarray(
                response,
                dtype=np.float32
            )

            # Some providers return a single vector
            # without the batch dimension.
            if vectors.ndim == 1 and len(batch) == 1:
                vectors = vectors.reshape(1, -1)

            if vectors.shape != (
                len(batch),
                EMBEDDING_DIMENSION
            ):
                raise ValueError(
                    "Unexpected embedding shape: "
                    f"{vectors.shape}. Expected "
                    f"({len(batch)}, {EMBEDDING_DIMENSION}). "
                    "Check the provider's pooling and model."
                )

            if not np.isfinite(vectors).all():
                raise ValueError(
                    "Embedding contains NaN or infinity."
                )

            if normalize_embeddings:
                norms = np.linalg.norm(
                    vectors,
                    axis=1,
                    keepdims=True
                )

                if np.any(norms == 0):
                    raise ValueError(
                        "The API returned a zero embedding."
                    )

                vectors = vectors / norms

            all_embeddings.append(vectors)

            if show_progress_bar:
                completed = min(
                    start + len(batch),
                    len(texts)
                )
                print(
                    f"Embedded {completed}/{len(texts)}"
                )

        result = np.vstack(all_embeddings)

        if single_input:
            return result[0]

        return result
