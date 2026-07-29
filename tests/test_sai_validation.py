from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sai_validation import (
    MedicalAgentBatchPredictor,
    MedicalAgentHttpPredictor,
    TaskSpec,
    _build_web_search_query,
    binary_metrics,
    build_parser,
    evaluate_task,
    load_task_records,
    parse_prediction_response,
)


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, spec, records):
        self.calls += 1
        return {
            int(record["excel_row"]): int(record["features"]["score"] >= 0.5)
            for record in records
        }


class SaiValidationTests(unittest.TestCase):
    def test_medical_agent_predictor_uses_full_hard_path(self):
        calls = []

        def fake_process(question, client, **kwargs):
            calls.append((question, client, kwargs))
            return '{"predictions":[{"excel_row":2,"prediction":1}]}', {}

        with tempfile.TemporaryDirectory() as temp_dir_text:
            predictor = MedicalAgentBatchPredictor.__new__(MedicalAgentBatchPredictor)
            predictor.client = object()
            predictor.process_diff_query = fake_process
            predictor.attempts = 1
            predictor.retry_delay = 0
            predictor.audit_dir = Path(temp_dir_text)
            predictor.call_index = 0
            result = predictor.predict(
                TaskSpec("SAI", "SAI.xlsx", "SAI", "卒中相关感染"),
                [{"excel_row": 2, "features": {"NIHSS": 7}}],
            )

            self.assertEqual(result, {2: 1})
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0][2]["need_rag"])
            self.assertIn("final_response_instruction", calls[0][2])
            self.assertIn("专家招募", calls[0][0])
            self.assertEqual(len(list(Path(temp_dir_text).glob("*.json"))), 1)

    def test_enable_web_search_cli_and_topic_query_exclude_case_features(self):
        args = build_parser().parse_args(["--enable-web-search"])
        self.assertTrue(args.enable_web_search)

        query = _build_web_search_query(
            TaskSpec("SAI", "SAI.xlsx", "SAI", "卒中相关感染风险预测")
        )
        self.assertIn("卒中相关感染", query)
        self.assertNotIn("NIHSS", query)
        self.assertLessEqual(len(query), 400)

    def test_http_predictor_sends_web_flag_and_requires_success_event(self):
        class FakeSseResponse:
            status = 200

            def __init__(self, events):
                self.lines = [
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
                    for event in events
                ]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def __iter__(self):
                return iter(self.lines)

        events = [
            {
                "type": "agent_output",
                "output": {
                    "agentName": "联网检索系统",
                    "content": "## 联网检索完成",
                },
            },
            {
                "type": "final_result",
                "content": '{"predictions":[{"excel_row":2,"prediction":1}]}',
            },
            {"type": "complete"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir_text:
            predictor = MedicalAgentHttpPredictor.__new__(MedicalAgentHttpPredictor)
            predictor.server_url = "http://127.0.0.1:50042"
            predictor.attempts = 1
            predictor.retry_delay = 0
            predictor.timeout = 10
            predictor.enable_web_search = True
            predictor.audit_dir = Path(temp_dir_text)
            predictor.call_index = 0
            with patch(
                "sai_validation.urllib.request.urlopen",
                return_value=FakeSseResponse(events),
            ) as opener:
                result = predictor._call(
                    TaskSpec("SAI", "SAI.xlsx", "SAI", "卒中相关感染风险预测"),
                    [{"excel_row": 2, "features": {"NIHSS": 7}}],
                )

            self.assertIn('"prediction":1', result)
            request = opener.call_args.args[0]
            payload = json.loads(request.data.decode("utf-8"))
            self.assertTrue(payload["enableWebSearch"])
            self.assertIn("卒中相关感染", payload["webSearchQuery"])
            self.assertNotIn("NIHSS", payload["webSearchQuery"])
            audit = json.loads(next(Path(temp_dir_text).glob("*.json")).read_text("utf-8"))
            self.assertTrue(audit["response"]["web_search"]["applied"])

    def test_parse_prediction_response_accepts_fenced_object(self):
        result = parse_prediction_response(
            '```json\n{"predictions":[{"excel_row":2,"prediction":0},'
            '{"excel_row":3,"prediction":"1"}]}\n```',
            [2, 3],
        )
        self.assertEqual(result, {2: 0, 3: 1})

    def test_parse_prediction_response_rejects_missing_row(self):
        with self.assertRaises(ValueError):
            parse_prediction_response('[{"excel_row":2,"prediction":0}]', [2, 3])

    def test_binary_metrics(self):
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall_sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_excel_load_evaluate_and_resume(self):
        spec = TaskSpec("TEST", "test.xlsx", "label", "test endpoint")
        with tempfile.TemporaryDirectory() as temp_dir_text:
            temp_dir = Path(temp_dir_text)
            pd.DataFrame(
                {
                    "score": [0.9, 0.1, 0.8, 0.2],
                    "other": [1, 2, 3, 4],
                    "label": [1, 0, 1, 0],
                }
            ).to_excel(temp_dir / "test.xlsx", index=False)
            features, records = load_task_records(spec, temp_dir)
            self.assertEqual(features, ["score", "other"])
            self.assertEqual([record["excel_row"] for record in records], [2, 3, 4, 5])

            predictor = FakePredictor()
            metrics, details = evaluate_task(
                spec,
                features,
                records,
                predictor,
                temp_dir / "output",
                batch_size=2,
                bootstrap_samples=0,
                seed=1,
            )
            self.assertEqual(metrics["f1"], 1.0)
            self.assertEqual(len(details), 4)
            self.assertEqual(predictor.calls, 2)

            network_predictor = FakePredictor()
            network_predictor.enable_web_search = True
            with self.assertRaisesRegex(ValueError, "联网检索模式"):
                evaluate_task(
                    spec,
                    features,
                    records,
                    network_predictor,
                    temp_dir / "output",
                    batch_size=2,
                    bootstrap_samples=0,
                    seed=1,
                )

            resumed_predictor = FakePredictor()
            resumed_metrics, _ = evaluate_task(
                spec,
                features,
                records,
                resumed_predictor,
                temp_dir / "output",
                batch_size=2,
                bootstrap_samples=0,
                seed=1,
            )
            self.assertEqual(resumed_metrics["f1"], 1.0)
            self.assertEqual(resumed_predictor.calls, 0)


if __name__ == "__main__":
    unittest.main()
