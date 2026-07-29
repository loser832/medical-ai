import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WebEvidenceWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.multi_source = (PROJECT_ROOT / "multi_agent.py").read_text(
            encoding="utf-8"
        )
        cls.md_source = (PROJECT_ROOT / "md_agent.py").read_text(
            encoding="utf-8"
        )

    def test_web_evidence_is_not_concatenated_into_original_question(self):
        ast.parse(self.multi_source)
        ast.parse(self.md_source)

        self.assertNotIn("augment_question_with_web", self.multi_source)
        self.assertIn("search_web_for_question(", self.multi_source)
        self.assertIn("is_source_lookup_query(question)", self.multi_source)
        self.assertIn("web_evidence_context=web_evidence_context", self.md_source)
        self.assertIn("question=question,\n        agent_profiles=", self.md_source)

    def test_runtime_status_endpoint_and_sse_status_are_wired(self):
        self.assertIn('@app.get("/chat/status/{session_id}")', self.multi_source)
        self.assertIn('"type": "status"', self.multi_source)
        self.assertIn('"workerAlive": worker_alive', self.multi_source)
        self.assertIn('"workerStoppedUnexpectedly"', self.multi_source)
        self.assertIn('"possiblyStalled"', self.multi_source)


if __name__ == "__main__":
    unittest.main()
