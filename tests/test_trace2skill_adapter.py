from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trace2skill_adapter.prepare_records import load_reviewed_cases
from trace2skill_adapter.recorder import TraceRecorder
from trace2skill_adapter.skill_loader import inject_skill, load_medical_skill
from trace2skill_adapter.validator import validate_skill_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = PROJECT_ROOT / "skills" / "medical-multi-agent"


class SkillValidationTests(unittest.TestCase):
    def test_checked_in_skill_is_valid(self):
        self.assertEqual(validate_skill_dir(SKILL_DIR), [])

    def test_loader_includes_direct_references_and_version(self):
        skill = load_medical_skill(SKILL_DIR)
        self.assertEqual(skill.name, "medical-multi-agent")
        self.assertEqual(len(skill.version), 12)
        self.assertIn("证据整合与专家共识", skill.content)
        self.assertIn("安全检查与风险升级", skill.content)
        prompt = inject_skill("基础角色", "主智能体", skill)
        self.assertIn(skill.version, prompt)
        self.assertIn("<medical_skill", prompt)


class TraceRecorderTests(unittest.TestCase):
    def test_hash_only_mode_does_not_persist_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                output_dir=temp_dir,
                session_id="patient-session",
                question="患者手机号 13800138000",
                model="test-model",
                capture_content=False,
            )
            recorder.record_agent_call(
                role="主智能体",
                prompt="原始提示",
                response="原始回答",
                latency_ms=12,
                status="success",
            )
            target = recorder.finalize("success", final_answer="最终回答")
            payload = target.read_text(encoding="utf-8")
            self.assertNotIn("原始提示", payload)
            self.assertNotIn("原始回答", payload)
            self.assertNotIn("13800138000", payload)
            parsed = json.loads(payload)
            self.assertEqual(parsed["capture_mode"], "hash_only")
            self.assertEqual(parsed["agent_calls"][0]["status"], "success")

    def test_content_mode_redacts_high_confidence_identifiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(
                output_dir=temp_dir,
                session_id="session",
                question="电话 13800138000 邮箱 patient@example.com 身份证 11010519491231002X",
                model="test-model",
                capture_content=True,
            )
            target = recorder.finalize("success")
            payload = target.read_text(encoding="utf-8")
            self.assertIn("[REDACTED_PHONE]", payload)
            self.assertIn("[REDACTED_EMAIL]", payload)
            self.assertIn("[REDACTED_PRC_ID]", payload)
            self.assertNotIn("patient@example.com", payload)


class ReviewedRecordTests(unittest.TestCase):
    def test_example_reviewed_case_converts_to_upstream_schema(self):
        source = PROJECT_ROOT / "data" / "trace2skill" / "reviewed_cases.example.jsonl"
        records = load_reviewed_cases(source)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["instance_id"], "medical-case-demo-001")
        self.assertEqual(
            {item["type"] for item in records[0]["items"]},
            {"failure_cause", "failure_memory"},
        )


if __name__ == "__main__":
    unittest.main()
