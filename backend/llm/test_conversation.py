import unittest
from unittest.mock import Mock

from backend.llm.conversation import quick_intent, route_conversation


class ConversationTests(unittest.TestCase):
    def test_social_messages_skip_model(self):
        query = Mock()
        for message in ['Hi!', 'Hello', 'HEY THERE', 'Good morning', 'How are you?',
                        'Thanks!', 'Bye', 'What can you do?']:
            with self.subTest(message=message):
                result = route_conversation(message, [], query, 'test')
                self.assertEqual(result['status'], 'conversational')
                self.assertEqual(result['retrieval'], [])
                self.assertEqual(result['sources'], [])
        query.assert_not_called()

    def test_mixed_messages_do_not_short_circuit(self):
        for message in ['Hi, how do I brace a wall?', 'Thanks, explain point two',
                        'What is a concrete slab?', 'Why?']:
            self.assertIsNone(quick_intent(message))

    def test_construction_and_followups_continue_to_retrieval(self):
        query = Mock(return_value={'answer': '{"intent":"construction"}'})
        history = [{'role': 'user', 'content': 'Explain slab inspections'}]
        self.assertIsNone(route_conversation('Why is that needed?', history, query, 'test'))
        self.assertIn('Explain slab inspections', query.call_args.kwargs['model_input']['user'])

    def test_general_questions_get_scope_response(self):
        query = Mock(return_value={'answer': '{"intent":"off_topic"}'})
        result = route_conversation('What is the weather?', [], query, 'test')
        self.assertEqual(result['sources'], [])
        self.assertIn('construction', result['answer'])

    def test_invalid_routing_asks_for_clarification(self):
        for answer in ['invalid', '[]', '{"intent":[]}', '{"intent":"invented"}']:
            query = Mock(return_value={'answer': answer})
            self.assertIn('little more', route_conversation('Something', [], query, 'test')['answer'])


if __name__ == '__main__':
    unittest.main()
