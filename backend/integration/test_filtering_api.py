"""Exercise 1A question/context requests through the real FastAPI route."""
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.main import app
from backend.integration.team2a import Team2AClient


class FilteringApiTests(unittest.TestCase):
    def setUp(self):
        self.rag = MagicMock()
        self.rag.answer_question.return_value = {
            "answer": "Record the outcome [MOCK-KO-001].", "status": "answered",
            "sources": ["MOCK-KO-001"],
            "retrieval": [{"ko_id": "MOCK-KO-001", "title": "Record", "score": 0.9}],
        }
        app.state.rag = self.rag
        app.state.embedding_model = MagicMock()
        app.state.eligibility_client = Team2AClient(eligibility_url="https://example.invalid/filter")
        app.state.filtering_enabled = False
        # No lifespan: real API validation/serialization, no model download or DB.
        self.client = TestClient(app)
        self.payload = {
            "question": "What should I check?",
            "context": {"module_id": "module-1", "stage_id": "stage-1",
                        "siteseer_node_id": "node-1", "selected_node_ids": ["node-1"],
                        "jurisdiction": "SA", "trade_or_domain": "Carpentry",
                        "learner_level": "Foundation"},
        }

    def tearDown(self):
        self.client.close()

    def test_off_accepts_1a_context_without_upstream_configuration(self):
        app.state.eligibility_client.eligibility_url = None
        app.state.filtering_enabled = False
        result = self.client.post("/ask", json=self.payload)
        self.assertEqual(result.status_code, 200)
        kwargs = self.rag.answer_question.call_args.kwargs
        self.assertIsNone(kwargs["eligibility_client"])
        self.assertEqual(kwargs["query_context"], self.payload["context"])
        self.assertNotIn("use_2a_filtering", kwargs)
        self.rag.get_database_connection.return_value.close.assert_called_once()

    def test_on_passes_context_and_client(self):
        app.state.filtering_enabled = True
        result = self.client.post("/ask", json=self.payload)
        self.assertEqual(result.status_code, 200)
        kwargs = self.rag.answer_question.call_args.kwargs
        self.assertIs(kwargs["eligibility_client"], app.state.eligibility_client)
        self.assertEqual(kwargs["query_context"], self.payload["context"])

    def test_on_without_service_returns_actionable_error_before_db(self):
        app.state.eligibility_client.eligibility_url = None
        app.state.filtering_enabled = True
        result = self.client.post("/ask", json=self.payload)
        self.assertEqual(result.status_code, 503)
        self.assertIn("TEAM2A_MODE=disabled", result.json()["detail"])
        self.rag.get_database_connection.assert_not_called()
        self.rag.answer_question.assert_not_called()

    def test_server_setting_controls_filtering(self):
        for enabled in (False, True):
            app.state.filtering_enabled = enabled
            result = self.client.post("/ask", json=self.payload)
            self.assertEqual(result.status_code, 200)
            actual = self.rag.answer_question.call_args.kwargs["eligibility_client"]
            self.assertEqual(actual is not None, enabled)

    def test_1a_context_required_in_both_modes(self):
        for enabled in (False, True):
            app.state.filtering_enabled = enabled
            for context in (None, {}):
                result = self.client.post("/ask", json={"question": "q", "context": context})
                self.assertEqual(result.status_code, 422)
            self.assertEqual(self.client.post("/ask", json={"question": "q"}).status_code, 422)
            for field in self.payload["context"]:
                context = {k: v for k, v in self.payload["context"].items() if k != field}
                self.assertEqual(self.client.post("/ask", json={"question": "q", "context": context}).status_code, 422)
        self.rag.answer_question.assert_not_called()

    def test_request_cannot_override_environment(self):
        for enabled in (False, True):
            app.state.filtering_enabled = enabled
            for requested in (False, True):
                result = self.client.post("/ask", json={**self.payload, "use_2a_filtering": requested})
                self.assertEqual(result.status_code, 422)
        self.rag.answer_question.assert_not_called()


if __name__ == "__main__":
    unittest.main()
