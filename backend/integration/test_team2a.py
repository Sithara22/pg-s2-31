"""Offline integration tests, with dependency stubs if the ML stack is absent."""
import copy
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from backend.integration.team2a import Team2AClient, Team2AError, map_bulk_snapshot


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((Path(__file__).resolve().parents[2] / "data/mock/team2a_bulk.json").read_text())

    def test_mapping_and_exclusion(self):
        self.assertEqual(map_bulk_snapshot(self.payload)[0]["ko_id"], "EKO-001-v1")
        for key, value in [("ai_indexable", False), ("viewer_exposable", False),
                           ("is_current", False), ("superseded", True),
                           ("publication_status", "draft"), ("validation_status", "unvalidated")]:
            payload = copy.deepcopy(self.payload)
            payload["ekos"][0][key] = value
            with self.subTest(key=key), self.assertRaises(Team2AError):
                map_bulk_snapshot(payload)

    def test_incomplete_and_duplicate_versions_rejected(self):
        self.payload["complete"] = False
        with self.assertRaises(Team2AError):
            map_bulk_snapshot(self.payload)
        self.payload["complete"] = True
        second = copy.deepcopy(self.payload["ekos"][0])
        second["ko_id"] = "EKO-001-v2"
        second["version"] = 2
        self.payload["ekos"].append(second)
        with self.assertRaises(Team2AError):
            map_bulk_snapshot(self.payload)

    def test_eligibility_request_and_invalid_responses(self):
        client = Team2AClient()
        with patch.object(client, "_request", return_value={"eko_ids": []}) as request:
            self.assertEqual(client.eligible_ids("Resolved question", {"stage_id": "s1"}), [])
            request.assert_called_once_with(client.eligibility_url, {"stage_id": "s1", "question": "Resolved question"})
        for response in [{}, {"eko_ids": None}, {"eko_ids": [""]}, {"eko_ids": ["a", "a"]}]:
            with patch.object(client, "_request", return_value=response), self.assertRaises(Team2AError):
                client.eligible_ids("q", {})

    def test_http_transport(self):
        client = Team2AClient(eligibility_url="https://example.invalid/filter", token="test-token")
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"eko_ids": ["EKO-001-v1"]}'
        with patch("backend.integration.team2a.urlopen", return_value=response) as http:
            self.assertEqual(client.eligible_ids("q", {}), ["EKO-001-v1"])
            request = http.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        with patch("backend.integration.team2a.urlopen", side_effect=TimeoutError), self.assertRaises(Team2AError):
            client.eligible_ids("q", {})

    def test_refresh_commits_only_after_verification_and_rolls_back_failure(self):
        from backend.integration.ingest import refresh_index
        records = map_bulk_snapshot(self.payload)
        indexer = types.ModuleType("backend.semantic_search.index_pgvector")
        connection = MagicMock()
        for name in ("create_table", "index_ekos", "validate_index_inputs", "verify_index"):
            setattr(indexer, name, MagicMock())
        indexer.get_database_connection = MagicMock(return_value=connection)
        ids = MagicMock()
        ids.tolist.return_value = ["EKO-001-v1"]
        modules = {
            "numpy": types.SimpleNamespace(array=MagicMock(return_value=ids)),
            "backend.semantic_search.generate_embeddings": types.SimpleNamespace(generate_embeddings=MagicMock()),
            "backend.semantic_search.index_pgvector": indexer,
        }
        with patch.dict(sys.modules, modules):
            refresh_index(records)
            connection.commit.assert_called_once()
            connection.close.assert_called_once()
            connection.reset_mock()
            indexer.verify_index.side_effect = ValueError("verification failed")
            with self.assertRaises(ValueError):
                refresh_index(records)
            connection.commit.assert_not_called()
            connection.rollback.assert_called_once()
            connection.close.assert_called_once()

    def test_invalid_snapshot_never_connects(self):
        self.payload["ekos"][0]["ai_indexable"] = "true"
        with self.assertRaises(Team2AError):
            map_bulk_snapshot(self.payload)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise real retrieval/orchestration logic without loading model weights.
        cls.stubs = patch.dict(sys.modules, {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=MagicMock()),
            "psycopg": types.ModuleType("psycopg"),
            "dotenv": types.SimpleNamespace(load_dotenv=lambda *a: None),
            "pgvector": types.ModuleType("pgvector"),
            "pgvector.psycopg": types.SimpleNamespace(register_vector=lambda c: None),
        })
        cls.stubs.start()
        cls.search = importlib.import_module("backend.semantic_search.search_pgvector")
        cls.rag = importlib.import_module("backend.llm.rag_pipeline")

    @classmethod
    def tearDownClass(cls):
        cls.stubs.stop()
        sys.modules.pop("backend.semantic_search.search_pgvector", None)
        sys.modules.pop("backend.llm.rag_pipeline", None)

    def test_empty_allowlist_does_not_embed_or_query(self):
        model, connection = MagicMock(), MagicMock()
        self.assertEqual(self.search.semantic_search_pgvector("q", model, connection, eligible_ko_ids=[]), [])
        model.encode.assert_not_called()
        connection.cursor.assert_not_called()

    def test_allowlist_is_parameterised_before_limit(self):
        model, connection = MagicMock(), MagicMock()
        connection.cursor.return_value.__enter__.return_value.fetchall.return_value = []
        self.search.semantic_search_pgvector("q", model, connection, stage="Slab", eligible_ko_ids=["EKO-001-v1"])
        sql, params = connection.cursor.return_value.__enter__.return_value.execute.call_args.args
        self.assertIn("ko_id = ANY(%s::text[])", sql)
        self.assertIn("AND stage = %s", sql)
        self.assertEqual(params[1:3], [["EKO-001-v1"], "Slab"])
        self.assertLess(sql.index("WHERE"), sql.index("ORDER BY"))

    def test_upstream_failure_never_runs_search(self):
        client = MagicMock()
        client.eligible_ids.side_effect = Team2AError("offline")
        with patch.object(self.rag, "semantic_search_pgvector") as search, self.assertRaises(Team2AError):
            self.rag.answer_question("q", MagicMock(), MagicMock(), eligibility_client=client)
        search.assert_not_called()

    def test_resolved_question_and_allowlist_reach_search(self):
        client = MagicMock()
        client.eligible_ids.return_value = ["EKO-001-v1"]
        with patch.object(self.rag, "resolve_question", return_value="resolved"), patch.object(
            self.rag, "semantic_search_pgvector", return_value=[]
        ) as search:
            result = self.rag.answer_question("that?", MagicMock(), MagicMock(), eligibility_client=client,
                                              query_context={"stage_id": "s1"})
        client.eligible_ids.assert_called_once_with("resolved", {"stage_id": "s1"})
        self.assertEqual(search.call_args.kwargs["eligible_ko_ids"], ["EKO-001-v1"])
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_real_version_id_citations(self):
        self.rag.validate_citations("Record it [EKO-001-v1].", {"EKO-001-v1"})
        with self.assertRaises(ValueError):
            self.rag.validate_citations("Record it [EKO-999-v1].", {"EKO-001-v1"})
        with self.assertRaises(ValueError):
            self.rag.validate_citations("Record it EKO-001-v1.", {"EKO-001-v1"})

    def test_answer_returns_non_mock_cited_sources(self):
        client = MagicMock()
        client.eligible_ids.return_value = ["EKO-001-v1"]
        evidence = [{"ko_id": "EKO-001-v1", "title": "Record", "score": 0.9, "searchable_text": "Record it."}]
        with patch.object(self.rag, "semantic_search_pgvector", return_value=evidence), patch.object(
            self.rag, "query_ollama", return_value={"answer": '{"status":"answered","answer":"Record it [EKO-001-v1]."}'}
        ):
            result = self.rag.answer_question("q", MagicMock(), MagicMock(), eligibility_client=client)
        self.assertEqual(result["sources"], ["EKO-001-v1"])
        self.assertEqual(result["status"], "answered")


if __name__ == "__main__":
    unittest.main()
