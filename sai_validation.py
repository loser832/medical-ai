"""Batch validation of SAI test sets through the current medical-agent system.

Each batch follows the project's full hard-query path: expert recruitment,
master-agent planning, RAG, expert discussion, and final decision. Predictions
are checkpointed as JSONL so a long validation run can resume after interruption.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

from utils import strip_thinking_content
from web_search import augment_question_with_web, search_web


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    os.getenv(
        "SAI_TEST_DATA_DIR",
        r"D:\xwechat_files\wxid_1e60rpai90l922_b64c\msg\file\2026-07",
    )
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sai_validation" / "medical_agent"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    filename: str
    label_column: str
    description: str


TASK_SPECS = {
    "A-SAI": TaskSpec(
        name="A-SAI",
        filename="A-SAI.xlsx",
        label_column="A-SAI",
        description=(
            "根据入院基线特征预测患者住院期间是否发生 A-SAI 亚型；"
            "A-SAI 的具体医学定义以研究方案为准"
        ),
    ),
    "NA-SAI": TaskSpec(
        name="NA-SAI",
        filename="NA-SAI.xlsx",
        label_column="NA-SAI",
        description=(
            "根据入院基线特征预测患者住院期间是否发生 NA-SAI 亚型；"
            "NA-SAI 的具体医学定义以研究方案为准"
        ),
    ),
    "SAI": TaskSpec(
        name="SAI",
        filename="SAI.xlsx",
        label_column="SAI",
        description="根据入院基线特征预测患者住院期间是否发生卒中相关感染（SAI）",
    ),
}


class PredictionFormatError(ValueError):
    """Raised when a model response cannot be parsed into complete 0/1 labels."""


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _feature_hash(features: dict[str, Any]) -> str:
    payload = json.dumps(features, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DOCUMENT_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if not match:
        raise ValueError(f"无效的 Excel 单元格地址：{cell_reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_scalar(cell: ElementTree.Element, shared_strings: Sequence[str]) -> Any:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{{{_SPREADSHEET_NS}}}t")
        )
    value_node = cell.find(f"{{{_SPREADSHEET_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type == "b":
        return 1 if raw == "1" else 0
    if cell_type in {"str", "e"}:
        return raw
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def _read_first_xlsx_sheet(path: Path) -> list[tuple[int, dict[int, Any]]]:
    """Read the first worksheet using only the Python standard library."""
    import posixpath

    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{_SPREADSHEET_NS}}}si"):
                shared_strings.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{_SPREADSHEET_NS}}}t")
                    )
                )

        workbook_root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook_root.find(
            f"{{{_SPREADSHEET_NS}}}sheets/{{{_SPREADSHEET_NS}}}sheet"
        )
        if first_sheet is None:
            raise ValueError(f"{path.name} 不包含工作表")
        relationship_id = first_sheet.get(f"{{{_DOCUMENT_REL_NS}}}id")
        relationships_root = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            relationship.get("Id"): relationship.get("Target")
            for relationship in relationships_root.findall(
                f"{{{_PACKAGE_REL_NS}}}Relationship"
            )
        }
        target = targets.get(relationship_id)
        if not target:
            raise ValueError(f"{path.name} 无法定位首个工作表")
        worksheet_path = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        worksheet_root = ElementTree.fromstring(archive.read(worksheet_path))

        rows: list[tuple[int, dict[int, Any]]] = []
        for row_node in worksheet_root.findall(
            f".//{{{_SPREADSHEET_NS}}}sheetData/{{{_SPREADSHEET_NS}}}row"
        ):
            row_number = int(row_node.get("r") or len(rows) + 1)
            values: dict[int, Any] = {}
            for cell in row_node.findall(f"{{{_SPREADSHEET_NS}}}c"):
                reference = cell.get("r")
                if not reference:
                    continue
                values[_column_index(reference)] = _xlsx_scalar(cell, shared_strings)
            if values:
                rows.append((row_number, values))
        return rows


def load_task_records(
    spec: TaskSpec,
    data_dir: Path,
    limit: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    path = data_dir / spec.filename
    if not path.is_file():
        raise FileNotFoundError(f"找不到测试集：{path}")

    rows = _read_first_xlsx_sheet(path)
    if not rows:
        raise ValueError(f"{path.name} 没有数据")
    _, header_cells = rows[0]
    header_pairs = [
        (column, str(value).strip())
        for column, value in sorted(header_cells.items())
        if value is not None and str(value).strip()
    ]
    headers = [header for _, header in header_pairs]
    if spec.label_column not in headers:
        raise ValueError(
            f"{path.name} 缺少标签列 {spec.label_column!r}；实际列为 {headers!r}"
        )
    duplicated = sorted({header for header in headers if headers.count(header) > 1})
    if duplicated:
        raise ValueError(f"{path.name} 存在重复列名：{duplicated!r}")

    column_by_header = {header: column for column, header in header_pairs}
    feature_columns = [header for header in headers if header != spec.label_column]
    if not feature_columns:
        raise ValueError(f"{path.name} 没有可用于推理的特征列")

    records: list[dict[str, Any]] = []
    data_rows = rows[1 : limit + 1 if limit is not None else None]
    missing_counts = {column: 0 for column in feature_columns}
    invalid_labels = set()
    for excel_row, row in data_rows:
        features = {}
        for column in feature_columns:
            value = _json_scalar(row.get(column_by_header[column]))
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_counts[column] += 1
            features[column] = value
        label_value = row.get(column_by_header[spec.label_column])
        try:
            numeric_label = float(label_value)
            label = int(numeric_label)
        except (TypeError, ValueError):
            invalid_labels.add(label_value)
            continue
        if numeric_label != label or label not in {0, 1}:
            invalid_labels.add(label_value)
            continue
        records.append(
            {
                "excel_row": excel_row,
                "features": features,
                "feature_hash": _feature_hash(features),
                "y_true": label,
            }
        )
    missing = {column: count for column, count in missing_counts.items() if count}
    if missing:
        raise ValueError(
            f"{path.name} 存在缺失特征 {missing!r}。测试阶段必须复用训练期填补规则。"
        )
    if invalid_labels:
        raise ValueError(f"{path.name} 标签必须为 0/1，发现：{sorted(map(str, invalid_labels))!r}")
    return feature_columns, records


def _candidate_json_values(text: str) -> Iterable[Any]:
    cleaned = strip_thinking_content(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        yield json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        yield value


def parse_prediction_response(text: str, expected_rows: Sequence[int]) -> dict[int, int]:
    expected = {int(row) for row in expected_rows}
    last_error = "未找到 JSON"

    for value in _candidate_json_values(text):
        if isinstance(value, dict):
            value = value.get("predictions", value.get("results"))
        if not isinstance(value, list):
            last_error = "JSON 顶层不是数组或 predictions/results 对象"
            continue

        parsed: dict[int, int] = {}
        try:
            for item in value:
                if not isinstance(item, dict):
                    raise PredictionFormatError("数组元素不是对象")
                row_value = item.get("excel_row", item.get("row_id", item.get("id")))
                prediction_value = item.get("prediction", item.get("pred", item.get("label")))
                row_id = int(row_value)
                if isinstance(prediction_value, bool):
                    prediction = int(prediction_value)
                elif str(prediction_value).strip() in {"0", "1"}:
                    prediction = int(str(prediction_value).strip())
                else:
                    raise PredictionFormatError(
                        f"第 {row_id} 行的预测不是 0/1：{prediction_value!r}"
                    )
                if row_id in parsed:
                    raise PredictionFormatError(f"第 {row_id} 行重复出现")
                parsed[row_id] = prediction
        except (TypeError, ValueError, PredictionFormatError) as error:
            last_error = str(error)
            continue

        missing = expected - set(parsed)
        extras = set(parsed) - expected
        if missing or extras:
            last_error = f"行号不完整；缺少={sorted(missing)!r}，多出={sorted(extras)!r}"
            continue
        return parsed

    preview = strip_thinking_content(text).replace("\n", " ")[:500]
    raise PredictionFormatError(f"无法解析完整预测：{last_error}；响应片段={preview!r}")


def _build_medical_agent_request(
    spec: TaskSpec,
    records: Sequence[dict[str, Any]],
) -> tuple[str, str]:
    cases = [
        {"excel_row": record["excel_row"], "features": record["features"]}
        for record in records
    ]
    question = (
        "这是一次回顾性医学科研测试，不是面向患者的诊疗请求。请让当前医疗多智能体"
        "系统完整执行专家招募、任务拆分、知识检索、专家讨论和最终决策。"
        "各病例必须相互独立判断，不得根据同批其他病例推断患病率。\n\n"
        f"目标终点：{spec.description}\n"
        "判定规则：1 表示目标终点阳性，0 表示阴性。真实标签未提供给系统。\n"
        "这是未来结局/住院期风险预测，不是判断患者在入院当下是否已存在感染。"
        "不得仅因未提供发热、CRP、PCT、培养或感染影像证据就直接判为0；"
        "应依据给出的基线危险因素综合预测终点发生风险，并为每例作出明确判断。\n"
        "最终必须为每个 excel_row 给出一个且仅一个 0/1 结论。\n\n"
        "待评估病例：\n"
        + json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    )
    final_instruction = (
        "在“最终回答”中输出 JSON 对象，格式为 "
        '{"predictions":[{"excel_row":2,"prediction":0}]}。'
        "prediction 只能是整数 0 或 1；必须覆盖输入中的每个 excel_row，"
        "不能增加其他行号。JSON 前后可以保留系统要求的一致性说明，但不得在 JSON 中"
        "加入解释性字段。"
    )
    return question, final_instruction


def _build_web_search_query(spec: TaskSpec) -> str:
    """Build a topic-only query; never send per-patient spreadsheet values."""
    return (
        f"{spec.name} {spec.description} "
        "危险因素 风险预测 临床研究 系统综述 指南"
    )[:400]


class MedicalAgentBatchPredictor:
    """Run a batch through the project's full hard medical-agent workflow."""

    def __init__(
        self,
        output_dir: Path,
        attempts: int = 1,
        retry_delay: float = 2.0,
        enable_web_search: bool = False,
    ):
        # Keep dry-runs and metric tests independent of the online client and
        # heavyweight retriever/model initialization.
        from llm_client import create_llm_client

        self.client = create_llm_client()
        from md_agent import process_diff_query

        self.process_diff_query = process_diff_query
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.enable_web_search = enable_web_search
        self.audit_dir = output_dir / "agent_audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0

    def close(self) -> None:
        self.client.close()

    def _call(self, spec: TaskSpec, records: Sequence[dict[str, Any]]) -> str:
        question, final_instruction = _build_medical_agent_request(spec, records)
        web_search_audit = {
            "requested": bool(getattr(self, "enable_web_search", False)),
            "applied": False,
            "provider": None,
            "query": None,
            "sources": [],
        }
        if web_search_audit["requested"]:
            search_query = _build_web_search_query(spec)
            web_response = search_web(search_query)
            question = augment_question_with_web(question, web_response)
            web_search_audit.update(
                {
                    "applied": True,
                    "provider": web_response.provider,
                    "query": search_query,
                    "sources": [result.url for result in web_response.results],
                }
            )

        self.call_index += 1
        rows = [int(record["excel_row"]) for record in records]
        call_id = f"{self.call_index:06d}_{spec.name}_{min(rows)}_{max(rows)}"
        log_path = self.audit_dir / f"{call_id}.log"
        audit_path = self.audit_dir / f"{call_id}.json"
        with log_path.open("w", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                final_decision, _ = self.process_diff_query(
                    question,
                    self.client,
                    callback=None,
                    need_rag=True,
                    final_response_instruction=final_instruction,
                )

        audit_path.write_text(
            json.dumps(
                {
                    "task": spec.name,
                    "excel_rows": rows,
                    "question": question,
                    "final_decision": final_decision,
                    "web_search": web_search_audit,
                    "full_log": str(log_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return final_decision or ""

    def predict(self, spec: TaskSpec, records: Sequence[dict[str, Any]]) -> dict[int, int]:
        expected_rows = [int(record["excel_row"]) for record in records]
        errors: list[str] = []
        format_failures = 0
        for attempt in range(1, self.attempts + 1):
            try:
                return parse_prediction_response(self._call(spec, records), expected_rows)
            except PredictionFormatError as error:
                format_failures += 1
                errors.append(f"第{attempt}次：{type(error).__name__}: {error}")
            except Exception as error:
                errors.append(f"第{attempt}次：{type(error).__name__}: {error}")
                if attempt >= self.attempts:
                    raise RuntimeError("医疗 Agent 连续运行失败：" + " | ".join(errors)) from error
            if attempt < self.attempts:
                time.sleep(self.retry_delay * attempt)

        # Split only malformed final output. Splitting an auth/transport failure
        # would multiply doomed full-agent runs exponentially.
        if format_failures == self.attempts and len(records) > 1:
            midpoint = len(records) // 2
            left = self.predict(spec, records[:midpoint])
            right = self.predict(spec, records[midpoint:])
            return {**left, **right}
        raise RuntimeError("医疗 Agent 预测结果无法解析：" + " | ".join(errors))


class MedicalAgentHttpPredictor(MedicalAgentBatchPredictor):
    """Use an already-running ``multi_agent.py`` service as the predictor."""

    def __init__(
        self,
        server_url: str,
        output_dir: Path,
        attempts: int = 1,
        retry_delay: float = 2.0,
        timeout: float = 3600.0,
        enable_web_search: bool = False,
    ):
        self.server_url = server_url.rstrip("/")
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_web_search = enable_web_search
        self.audit_dir = output_dir / "agent_audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.call_index = 0
        try:
            with urllib.request.urlopen(
                f"{self.server_url}/docs", timeout=10
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"无法连接已运行的医疗 Agent 服务 {self.server_url}：{error}"
            ) from error

    def close(self) -> None:
        return None

    def _call(self, spec: TaskSpec, records: Sequence[dict[str, Any]]) -> str:
        question, final_instruction = _build_medical_agent_request(spec, records)
        # The HTTP API does not expose process_diff_query's internal
        # output-contract argument, so carry the same contract in the question.
        question = f"{question}\n\n最终输出约束：{final_instruction}"
        session_id = (
            f"sai-eval-{os.getpid()}-{time.time_ns()}-{self.call_index + 1}"
        )
        request_payload = {
            "query": question,
            "id": session_id,
            "enableMultiAgent": True,
            "enableDifficultyAgent": False,
            "difficulty": "hard",
            "enableWebSearch": self.enable_web_search,
        }
        if self.enable_web_search:
            request_payload["webSearchQuery"] = _build_web_search_query(spec)
        request = urllib.request.Request(
            f"{self.server_url}/chat/stream",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        stream_events: list[dict[str, Any]] = []
        final_decision: str | None = None
        stream_error: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            f"医疗 Agent SSE 事件无法解析：{line[:500]}"
                        ) from error
                    if not isinstance(event, dict):
                        continue
                    stream_events.append(event)
                    event_type = event.get("type")
                    if event_type == "final_result":
                        content = event.get("content")
                        if isinstance(content, str) and content.strip():
                            final_decision = content
                    elif event_type == "error":
                        stream_error = str(event.get("error") or "未知流式错误")
                    elif event_type == "complete":
                        break
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"医疗 Agent HTTP {error.code}：{body}") from error
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(f"医疗 Agent HTTP 调用失败：{error}") from error

        if final_decision is None:
            suffix = f"；服务错误：{stream_error}" if stream_error else ""
            raise RuntimeError(
                "医疗 Agent SSE 响应缺少 final_result"
                f"（事件：{[event.get('type') for event in stream_events]}）{suffix}"
            )

        web_search_messages = [
            str(event.get("output", {}).get("content") or "")
            for event in stream_events
            if event.get("type") == "agent_output"
            and event.get("output", {}).get("agentName") == "联网检索系统"
        ]
        web_search_applied = any(
            "联网检索完成" in message for message in web_search_messages
        )
        web_search_message = web_search_messages[-1] if web_search_messages else None

        self.call_index += 1
        rows = [int(record["excel_row"]) for record in records]
        call_id = f"{self.call_index:06d}_{spec.name}_{min(rows)}_{max(rows)}"
        (self.audit_dir / f"{call_id}.json").write_text(
            json.dumps(
                {
                    "task": spec.name,
                    "excel_rows": rows,
                    "server_url": self.server_url,
                    "request": request_payload,
                    "response": {
                        "final_result": final_decision,
                        "event_types": [
                            event.get("type") for event in stream_events
                        ],
                        "stream_error": stream_error,
                        "web_search": {
                            "requested": self.enable_web_search,
                            "applied": web_search_applied,
                            "message": web_search_message,
                        },
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.enable_web_search and not web_search_applied:
            detail = web_search_message or "服务未返回联网检索状态"
            raise RuntimeError(f"已要求联网检索，但本批次未成功：{detail}")
        return final_decision


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true 与 y_pred 必须等长且非空")
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    def divide(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    accuracy = divide(tp + tn, len(y_true))
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "n": len(y_true),
        "positive": sum(int(value) for value in y_true),
        "prevalence": divide(sum(int(value) for value in y_true), len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def bootstrap_f1_interval(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if samples <= 0:
        return None
    rng = random.Random(seed)
    size = len(y_true)
    estimates = []
    for _ in range(samples):
        indexes = [rng.randrange(size) for _ in range(size)]
        truth = [y_true[index] for index in indexes]
        pred = [y_pred[index] for index in indexes]
        estimates.append(float(binary_metrics(truth, pred)["f1"]))
    estimates.sort()
    lower = estimates[max(0, int(0.025 * samples) - 1)]
    upper = estimates[min(samples - 1, int(0.975 * samples))]
    return lower, upper


def _load_checkpoint(
    path: Path,
    records_by_row: dict[int, dict[str, Any]],
    web_search_enabled: bool = False,
) -> dict[int, int]:
    predictions: dict[int, int] = {}
    if not path.is_file():
        return predictions
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            row = int(item["excel_row"])
            record = records_by_row.get(row)
            if record is None:
                continue
            if item.get("feature_hash") != record["feature_hash"]:
                raise ValueError(
                    f"断点文件 {path} 第 {line_number} 行与当前测试集第 {row} 行不一致。"
                    "请使用新的 --output-dir，避免复用旧预测。"
                )
            checkpoint_web_search = bool(item.get("web_search_enabled", False))
            if checkpoint_web_search != web_search_enabled:
                raise ValueError(
                    f"断点文件 {path} 的联网检索模式与当前命令不一致。"
                    "请使用新的 --output-dir，或显式使用 --no-resume 重新预测。"
                )
            predictions[row] = int(item["y_pred"])
    return predictions


def _chunks(values: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def evaluate_task(
    spec: TaskSpec,
    feature_columns: Sequence[str],
    records: Sequence[dict[str, Any]],
    predictor: Any,
    output_dir: Path,
    batch_size: int,
    bootstrap_samples: int,
    seed: int,
    resume: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{spec.name}_predictions.jsonl"
    records_by_row = {int(record["excel_row"]): record for record in records}
    web_search_enabled = bool(getattr(predictor, "enable_web_search", False))
    predictions = (
        _load_checkpoint(checkpoint_path, records_by_row, web_search_enabled)
        if resume
        else {}
    )
    pending = [record for record in records if int(record["excel_row"]) not in predictions]
    random.Random(seed).shuffle(pending)
    mode = "a" if resume else "w"

    print(
        f"[{spec.name}] 样本={len(records)}，已完成={len(predictions)}，待预测={len(pending)}，"
        f"批大小={batch_size}",
        flush=True,
    )
    with checkpoint_path.open(mode, encoding="utf-8", newline="\n") as checkpoint:
        completed_now = 0
        for batch in _chunks(pending, batch_size):
            batch_rows = [int(record["excel_row"]) for record in batch]
            print(
                f"[{spec.name}] 启动医疗 Agent 批次：rows={batch_rows}",
                flush=True,
            )
            batch_predictions = predictor.predict(spec, batch)
            for record in batch:
                row = int(record["excel_row"])
                prediction = int(batch_predictions[row])
                predictions[row] = prediction
                payload = {
                    "task": spec.name,
                    "excel_row": row,
                    "feature_hash": record["feature_hash"],
                    "web_search_enabled": web_search_enabled,
                    "y_true": int(record["y_true"]),
                    "y_pred": prediction,
                }
                checkpoint.write(json.dumps(payload, ensure_ascii=False) + "\n")
            checkpoint.flush()
            completed_now += len(batch)
            print(
                f"[{spec.name}] 本次进度 {completed_now}/{len(pending)}；总进度 "
                f"{len(predictions)}/{len(records)}",
                flush=True,
            )

    details = []
    for record in records:
        row = int(record["excel_row"])
        details.append(
            {
                "task": spec.name,
                "excel_row": row,
                "y_true": int(record["y_true"]),
                "y_pred": int(predictions[row]),
                "correct": int(record["y_true"]) == int(predictions[row]),
                "web_search_enabled": web_search_enabled,
            }
        )
    y_true = [item["y_true"] for item in details]
    y_pred = [item["y_pred"] for item in details]
    metrics = binary_metrics(y_true, y_pred)
    interval = bootstrap_f1_interval(y_true, y_pred, bootstrap_samples, seed)
    metrics.update(
        {
            "task": spec.name,
            "label_column": spec.label_column,
            "feature_columns": list(feature_columns),
            "web_search_enabled": web_search_enabled,
            "f1_ci95_low": interval[0] if interval else None,
            "f1_ci95_high": interval[1] if interval else None,
        }
    )
    return metrics, details


def _write_outputs(
    output_dir: Path,
    metrics: Sequence[dict[str, Any]],
    details: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(list(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_fields = [
        "task",
        "web_search_enabled",
        "n",
        "positive",
        "prevalence",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "f1_ci95_low",
        "f1_ci95_high",
        "accuracy",
        "balanced_accuracy",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    with (output_dir / "prediction_details.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "task",
            "excel_row",
            "y_true",
            "y_pred",
            "correct",
            "web_search_enabled",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用当前医疗多智能体系统验证 A-SAI、NA-SAI、SAI 测试集。"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--server-url",
        default=os.getenv("MEDICAL_AGENT_URL", "http://127.0.0.1:50042"),
        help="已运行的 multi_agent.py 服务地址",
    )
    parser.add_argument(
        "--local-agent",
        action="store_true",
        help="不调用HTTP服务，改为在当前进程直接加载医疗Agent",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help=(
            "为每个批次启用联网检索；只发送任务终点检索词，不发送逐例特征。"
            "联网失败时中止，避免联网与非联网预测混用"
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_SPECS),
        default=list(TASK_SPECS),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--task-description",
        action="append",
        default=[],
        metavar="TASK=DESCRIPTION",
        help="覆盖终点医学定义，可重复指定，例如 SAI=卒中相关感染",
    )
    parser.add_argument("--limit", type=int, help="每个任务仅取前 N 行，用于连通性测试")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        help="每个任务随机抽取 N 个阴性和 N 个阳性病例，用于平衡小样本测试",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查文件、字段和标签分布，不调用模型",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit 必须大于 0")
    if args.sample_per_class is not None and args.sample_per_class <= 0:
        raise SystemExit("--sample-per-class 必须大于 0")
    if args.limit is not None and args.sample_per_class is not None:
        raise SystemExit("--limit 与 --sample-per-class 不能同时使用")

    description_overrides = {}
    for item in args.task_description:
        task_name, separator, description = item.partition("=")
        if not separator or task_name not in TASK_SPECS or not description.strip():
            raise SystemExit(
                "--task-description 格式必须为 TASK=DESCRIPTION，"
                f"TASK 可选 {list(TASK_SPECS)!r}"
            )
        description_overrides[task_name] = description.strip()

    loaded = []
    for task_name in args.tasks:
        spec = TASK_SPECS[task_name]
        if task_name in description_overrides:
            spec = replace(spec, description=description_overrides[task_name])
        features, records = load_task_records(spec, args.data_dir, args.limit)
        if args.sample_per_class is not None:
            negatives = [record for record in records if int(record["y_true"]) == 0]
            positives = [record for record in records if int(record["y_true"]) == 1]
            if (
                len(negatives) < args.sample_per_class
                or len(positives) < args.sample_per_class
            ):
                raise SystemExit(
                    f"{task_name} 不足以各抽取 {args.sample_per_class} 个阴性和阳性病例"
                )
            task_rng = random.Random(f"{args.seed}:{task_name}")
            records = task_rng.sample(negatives, args.sample_per_class) + task_rng.sample(
                positives, args.sample_per_class
            )
        loaded.append((spec, features, records))
        positives = sum(int(record["y_true"]) for record in records)
        print(
            f"[{task_name}] 文件={args.data_dir / spec.filename}，样本={len(records)}，"
            f"阳性={positives}，特征={len(features)}",
            flush=True,
        )

    if args.dry_run:
        print("Dry-run 完成：未调用模型。")
        return 0

    try:
        if args.local_agent:
            predictor = MedicalAgentBatchPredictor(
                output_dir=args.output_dir,
                attempts=args.attempts,
                retry_delay=args.retry_delay,
                enable_web_search=args.enable_web_search,
            )
        else:
            predictor = MedicalAgentHttpPredictor(
                server_url=args.server_url,
                output_dir=args.output_dir,
                attempts=args.attempts,
                retry_delay=args.retry_delay,
                timeout=args.request_timeout,
                enable_web_search=args.enable_web_search,
            )
    except RuntimeError as error:
        print(f"无法启动医疗 Agent 验证：{error}", file=sys.stderr)
        return 2
    all_metrics = []
    all_details = []
    try:
        for offset, (spec, features, records) in enumerate(loaded):
            metrics, details = evaluate_task(
                spec=spec,
                feature_columns=features,
                records=records,
                predictor=predictor,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap,
                seed=args.seed + offset,
                resume=not args.no_resume,
            )
            all_metrics.append(metrics)
            all_details.extend(details)
            print(
                f"[{spec.name}] F1={metrics['f1']:.4f}，"
                f"Precision={metrics['precision'] if metrics['precision'] is not None else 'NA'}，"
                f"Recall={metrics['recall_sensitivity'] if metrics['recall_sensitivity'] is not None else 'NA'}",
                flush=True,
            )
    finally:
        predictor.close()

    _write_outputs(args.output_dir, all_metrics, all_details)
    print(f"验证完成，结果目录：{args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n验证已中断；已有 JSONL 预测已保存，下次默认断点续跑。", file=sys.stderr)
        raise SystemExit(130)
