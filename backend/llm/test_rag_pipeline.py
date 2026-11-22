"""Regression checks for local API startup and Ollama generation."""

import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.llm import rag_pipeline as rag


class LocalRagTests(unittest.TestCase):
    def test_api_lifespan_loads_local_embeddings(self):
        with patch.object(rag, "SentenceTransformer") as model:
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").json(), {"status": "ok"})
                self.assertIs(app.state.embedding_model, model.return_value)
            model.assert_called_once_with(rag.MODEL_NAME)

    def test_retrieved_evidence_is_answered_with_ollama(self):
        evidence = [{"ko_id": "MOCK-KO-001", "title": "Inspection",
                     "searchable_text": "Inspect before pouring.", "score": 0.9}]
        with patch.object(rag, "route_conversation", return_value=None), \
             patch.object(rag, "semantic_search_pgvector", return_value=evidence), \
             patch.object(rag, "query_ollama", return_value={
                 "answer": '{"status":"answered","answer":"Inspect before pouring [MOCK-KO-001]."}'
             }) as query:
            result = rag.answer_question("When should I inspect?", Mock(), Mock())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"], ["MOCK-KO-001"])
        self.assertEqual(query.call_args.kwargs["model"], rag.OLLAMA_MODEL)


if __name__ == "__main__":
    unittest.main()
