import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.main import UnansweredQuestionListResponse
from backend.unanswered_questions import list_unanswered_questions


class UnansweredQuestionsTests(unittest.TestCase):
    def test_legacy_history_does_not_break_listing_response(self):
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value.fetchall.return_value = [{
            "total": 1, "id": 1, "question": "Explain roofing",
            "topics": ["roofing"], "keywords": ["roofing"],
            "filters": {"history": [], "trade": "Roofing"},
            "reason": "insufficient_support", "created_at": datetime.now(timezone.utc),
        }]
        result = list_unanswered_questions(connection, limit=10, offset=0)
        response = UnansweredQuestionListResponse.model_validate(result)
        self.assertEqual(response.items[0].filters, {"trade": "Roofing"})
