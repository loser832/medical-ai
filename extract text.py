"""从 LangChain FAISS 的 index.pkl 中提取问答对并导出为 Excel。

注意：pickle 反序列化可能执行任意代码，只能处理可信来源的 ``index.pkl``。

示例：
    python "extract text.py"
    python "extract text.py" ./models/faiss_index_A_v4/index.pkl -o ./outputs/qa_pairs.xlsx
    python "extract text.py" --deduplicate --overwrite
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import xlsxwriter
except ImportError as exc:  # pragma: no cover - 取决于用户环境
    raise SystemExit(
        "缺少 xlsxwriter。请先运行：pip install -r requirements.txt"
    ) from exc


DEFAULT_PKL_PATH = Path("./models/faiss_index_A_v4/index.pkl")
DEFAULT_OUTPUT_PATH = Path("./outputs/qa_pairs.xlsx")
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_CELL_CHARS = 32_767
DEFAULT_ROWS_PER_SHEET = 1_000_000

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


class _CompatDocument:
    """在本机未安装 LangChain 时承接 pickle 中的 Document 状态。"""


class _CompatInMemoryDocstore:
    """在本机未安装 LangChain 时承接 InMemoryDocstore 状态。"""

    def search(self, doc_id: str) -> Any:
        return getattr(self, "_dict", {}).get(doc_id)


class _CompatibilityUnpickler(pickle.Unpickler):
    """只为当前索引涉及的两个 LangChain 类型提供兼容映射。"""

    _CLASS_MAP = {
        ("langchain_core.documents.base", "Document"): _CompatDocument,
        (
            "langchain_community.docstore.in_memory",
            "InMemoryDocstore",
        ): _CompatInMemoryDocstore,
    }

    def find_class(self, module: str, name: str) -> Any:
        mapped = self._CLASS_MAP.get((module, name))
        if mapped is not None:
            return mapped
        return super().find_class(module, name)


@dataclass(frozen=True)
class QAPair:
    index_number: Any
    doc_id: str
    question: str
    answer: str
    source: str
    line_number: Any


@dataclass
class ExportStats:
    input_rows: int = 0
    exported_rows: int = 0
    skipped_empty_rows: int = 0
    deduplicated_rows: int = 0
    truncated_cells: int = 0
    worksheet_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 FAISS index.pkl 提取问答对，规整后写入 .xlsx 文件。"
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PKL_PATH,
        help=f"index.pkl 路径（默认：{DEFAULT_PKL_PATH.as_posix()}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出 Excel 路径（默认：{DEFAULT_OUTPUT_PATH.as_posix()}）",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="按“问题 + 答案”去除完全重复的问答对；默认保留索引中的全部记录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件。",
    )
    parser.add_argument(
        "--rows-per-sheet",
        type=int,
        default=DEFAULT_ROWS_PER_SHEET,
        help=(
            "每个问答工作表的数据行数，超出后自动分表"
            f"（默认：{DEFAULT_ROWS_PER_SHEET}，最大：{EXCEL_MAX_ROWS - 1}）。"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只导出前 N 条有效问答；适合快速测试，默认导出全部。",
    )
    args = parser.parse_args()

    if not 1 <= args.rows_per_sheet <= EXCEL_MAX_ROWS - 1:
        parser.error(f"--rows-per-sheet 必须在 1 到 {EXCEL_MAX_ROWS - 1} 之间")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数")
    return args


def load_index(pkl_path: Path) -> tuple[Any, Mapping[Any, str]]:
    """加载索引；缺少 LangChain 时使用轻量兼容类读取已有字段。"""

    if not pkl_path.is_file():
        raise FileNotFoundError(f"找不到索引文件：{pkl_path}")

    try:
        with pkl_path.open("rb") as file:
            loaded = pickle.load(file)
    except (ModuleNotFoundError, ImportError, AttributeError):
        with pkl_path.open("rb") as file:
            loaded = _CompatibilityUnpickler(file).load()

    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("index.pkl 结构不符合预期：应包含 (docstore, index_to_docstore_id)")

    docstore, index_to_docstore_id = loaded
    if not isinstance(index_to_docstore_id, Mapping):
        raise ValueError("index_to_docstore_id 不是映射类型")
    return docstore, index_to_docstore_id


def normalize_text(value: Any) -> str:
    """清除 Excel 不支持的控制字符，并保留最多两段连续换行。"""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", text).strip()


def _document_fields(document: Any) -> tuple[str, str, str, Any]:
    """兼容真实 LangChain Document 与兼容反序列化后的嵌套状态。"""

    state = vars(document) if hasattr(document, "__dict__") else {}
    nested_state = state.get("__dict__")
    payload = nested_state if isinstance(nested_state, dict) else state

    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    page_content = getattr(document, "page_content", None)
    if page_content is None:
        page_content = payload.get("page_content")

    return (
        normalize_text(metadata.get("qa_question")),
        normalize_text(page_content),
        normalize_text(metadata.get("source")),
        metadata.get("line_num"),
    )


def iter_qa_pairs(
    docstore: Any, index_to_docstore_id: Mapping[Any, str]
) -> Iterator[QAPair]:
    documents = getattr(docstore, "_dict", None)
    search = getattr(docstore, "search", None)

    if not isinstance(documents, Mapping) and not callable(search):
        raise ValueError("docstore 既没有可用的 _dict，也没有 search 方法")

    for index_number, doc_id_value in index_to_docstore_id.items():
        doc_id = str(doc_id_value)
        document = (
            documents.get(doc_id_value)
            if isinstance(documents, Mapping)
            else search(doc_id_value)
        )
        if document is None:
            continue

        question, answer, source, line_number = _document_fields(document)
        yield QAPair(
            index_number=index_number,
            doc_id=doc_id,
            question=question,
            answer=answer,
            source=source,
            line_number=line_number,
        )


def fit_excel_cell(text: str) -> tuple[str, bool]:
    """限制文本长度，避免超过 Excel 单元格的 32,767 字符上限。"""

    if len(text) <= EXCEL_MAX_CELL_CHARS:
        return text, False
    suffix = "\n[内容超过 Excel 单元格上限，已截断]"
    return text[: EXCEL_MAX_CELL_CHARS - len(suffix)] + suffix, True


def wrapped_line_count(text: str, column_width: int) -> int:
    """按中英文字符的近似显示宽度估算 Excel 自动换行行数。"""

    lines = 0
    for paragraph in text.split("\n") or [""]:
        display_width = sum(
            2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            for char in paragraph
        )
        lines += max(1, (display_width + column_width - 1) // column_width)
    return lines


def estimate_qa_row_height(pair: QAPair) -> float:
    """让长问题和答案在打开 Excel 时尽量完整显示。"""

    line_count = max(
        wrapped_line_count(pair.question, 42),
        wrapped_line_count(pair.answer, 80),
        wrapped_line_count(pair.source, 45),
        wrapped_line_count(pair.doc_id, 38),
    )
    return min(409.0, max(30.0, line_count * 15.0 + 4.0))


def create_formats(workbook: Any) -> dict[str, Any]:
    return {
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "align": "center",
                "valign": "vcenter",
                "border": 0,
            }
        ),
        "text": workbook.add_format(
            {"font_color": "#1F2937", "valign": "top", "text_wrap": True}
        ),
        "center": workbook.add_format(
            {"font_color": "#374151", "align": "center", "valign": "top"}
        ),
        "integer": workbook.add_format(
            {
                "font_color": "#374151",
                "align": "right",
                "valign": "top",
                "num_format": "0",
            }
        ),
        "title": workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "section": workbook.add_format(
            {
                "bold": True,
                "font_color": "#1F4E78",
                "bg_color": "#D9EAF7",
                "align": "left",
                "valign": "vcenter",
            }
        ),
        "label": workbook.add_format(
            {"bold": True, "font_color": "#374151", "bg_color": "#F3F6F9"}
        ),
        "value": workbook.add_format({"font_color": "#111827"}),
        "note": workbook.add_format(
            {
                "font_color": "#6B7280",
                "italic": True,
                "text_wrap": True,
                "valign": "top",
            }
        ),
    }


def configure_qa_sheet(worksheet: Any, formats: Mapping[str, Any]) -> None:
    headers = ["序号", "问题", "答案", "来源", "原始行号", "文档ID", "索引编号"]
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(1, 0)
    worksheet.set_zoom(90)
    worksheet.set_row(0, 28)
    worksheet.set_column("A:A", 10)
    worksheet.set_column("B:B", 42)
    worksheet.set_column("C:C", 80)
    worksheet.set_column("D:D", 45)
    worksheet.set_column("E:E", 12)
    worksheet.set_column("F:F", 38)
    worksheet.set_column("G:G", 12)
    for column, header in enumerate(headers):
        worksheet.write_string(0, column, header, formats["header"])


def write_mixed_value(
    worksheet: Any,
    row: int,
    column: int,
    value: Any,
    text_format: Any,
    number_format: Any,
) -> None:
    if isinstance(value, bool):
        worksheet.write_boolean(row, column, value, text_format)
    elif isinstance(value, (int, float)):
        worksheet.write_number(row, column, value, number_format)
    elif value is None:
        worksheet.write_blank(row, column, None, text_format)
    else:
        worksheet.write_string(row, column, normalize_text(value), text_format)


def write_summary_sheet(
    worksheet: Any,
    formats: Mapping[str, Any],
    source_path: Path,
    output_path: Path,
    stats: ExportStats,
    source_counts: Counter[str],
    deduplicate: bool,
) -> None:
    worksheet.hide_gridlines(2)
    worksheet.set_zoom(95)
    worksheet.set_column("A:A", 24)
    worksheet.set_column("B:B", 78)
    worksheet.write_string("A1", "医学问答对导出说明", formats["title"])
    worksheet.write_blank("B1", None, formats["title"])
    worksheet.write_blank("A2", None, formats["title"])
    worksheet.write_blank("B2", None, formats["title"])
    worksheet.set_row(0, 26)
    worksheet.set_row(1, 10)

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    summary_rows = [
        ("生成时间", generated_at),
        ("源索引", str(source_path.resolve())),
        ("输出文件", str(output_path.resolve())),
        ("原始映射记录数", stats.input_rows),
        ("已导出问答数", stats.exported_rows),
        ("跳过空问答数", stats.skipped_empty_rows),
        ("去重记录数", stats.deduplicated_rows),
        ("是否启用去重", "是" if deduplicate else "否"),
        ("问答工作表数", stats.worksheet_count),
        ("截断单元格数", stats.truncated_cells),
    ]
    for row, (label, value) in enumerate(summary_rows, start=3):
        worksheet.write_string(row, 0, label, formats["label"])
        write_mixed_value(
            worksheet,
            row,
            1,
            value,
            formats["value"],
            formats["integer"],
        )

    source_header_row = len(summary_rows) + 5
    worksheet.write_string(source_header_row, 0, "来源文件", formats["section"])
    worksheet.write_string(source_header_row, 1, "问答数量", formats["section"])
    for row, (source, count) in enumerate(
        sorted(source_counts.items(), key=lambda item: (-item[1], item[0])),
        start=source_header_row + 1,
    ):
        worksheet.write_string(row, 0, source or "（未标注来源）", formats["text"])
        worksheet.write_number(row, 1, count, formats["integer"])

    end_row = source_header_row + max(1, len(source_counts))
    worksheet.autofilter(source_header_row, 0, end_row, 1)
    worksheet.write_string(
        end_row + 2,
        0,
        "说明",
        formats["section"],
    )
    worksheet.write_string(
        end_row + 3,
        0,
        "问答工作表按索引顺序写入；问题与答案已清理控制字符、统一换行和空白。"
        "为避免公式注入，所有文本均按普通字符串保存。",
        formats["note"],
    )
    worksheet.set_row(end_row + 3, 90)
    worksheet.write_blank(end_row + 3, 1, None, formats["note"])


def export_to_excel(
    source_path: Path,
    output_path: Path,
    docstore: Any,
    index_to_docstore_id: Mapping[Any, str],
    *,
    deduplicate: bool,
    rows_per_sheet: int,
    limit: int | None,
    overwrite: bool,
) -> ExportStats:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已存在：{output_path}。如需覆盖，请添加 --overwrite。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()

    stats = ExportStats(input_rows=len(index_to_docstore_id))
    source_counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] | None = set() if deduplicate else None

    workbook = xlsxwriter.Workbook(
        str(temporary_path),
        {
            "constant_memory": True,
            "strings_to_formulas": False,
            "strings_to_urls": False,
            "tmpdir": str(output_path.parent),
        },
    )
    workbook.set_properties(
        {
            "title": "医学问答对",
            "subject": "从 LangChain FAISS 索引提取的结构化问答数据",
            "author": "medical-ai",
            "comments": "由 extract text.py 自动生成",
        }
    )
    formats = create_formats(workbook)
    summary_sheet = workbook.add_worksheet("导出说明")

    qa_sheets: list[tuple[Any, int]] = []
    current_sheet = None
    current_excel_row = 1

    try:
        for pair in iter_qa_pairs(docstore, index_to_docstore_id):
            if not pair.question or not pair.answer:
                stats.skipped_empty_rows += 1
                continue

            if seen is not None:
                key = (pair.question, pair.answer)
                if key in seen:
                    stats.deduplicated_rows += 1
                    continue
                seen.add(key)

            if limit is not None and stats.exported_rows >= limit:
                break

            if current_sheet is None or current_excel_row > rows_per_sheet:
                if current_sheet is not None:
                    qa_sheets.append((current_sheet, current_excel_row - 1))
                sheet_number = len(qa_sheets) + 1
                current_sheet = workbook.add_worksheet(f"问答对_{sheet_number}")
                configure_qa_sheet(current_sheet, formats)
                current_excel_row = 1

            question, question_truncated = fit_excel_cell(pair.question)
            answer, answer_truncated = fit_excel_cell(pair.answer)
            source, source_truncated = fit_excel_cell(pair.source)
            stats.truncated_cells += sum(
                (question_truncated, answer_truncated, source_truncated)
            )

            row = current_excel_row
            current_sheet.set_row(row, estimate_qa_row_height(pair))
            current_sheet.write_number(row, 0, stats.exported_rows + 1, formats["integer"])
            current_sheet.write_string(row, 1, question, formats["text"])
            current_sheet.write_string(row, 2, answer, formats["text"])
            current_sheet.write_string(row, 3, source, formats["text"])
            write_mixed_value(
                current_sheet,
                row,
                4,
                pair.line_number,
                formats["center"],
                formats["integer"],
            )
            current_sheet.write_string(row, 5, pair.doc_id, formats["center"])
            write_mixed_value(
                current_sheet,
                row,
                6,
                pair.index_number,
                formats["center"],
                formats["integer"],
            )

            source_counts[pair.source] += 1
            stats.exported_rows += 1
            current_excel_row += 1

            if stats.exported_rows % 10_000 == 0:
                print(f"已写入 {stats.exported_rows:,} 条问答……")

        if current_sheet is None:
            current_sheet = workbook.add_worksheet("问答对_1")
            configure_qa_sheet(current_sheet, formats)
            qa_sheets.append((current_sheet, 0))
        else:
            qa_sheets.append((current_sheet, current_excel_row - 1))

        for worksheet, data_rows in qa_sheets:
            worksheet.autofilter(0, 0, data_rows, 6)

        stats.worksheet_count = len(qa_sheets)
        write_summary_sheet(
            summary_sheet,
            formats,
            source_path,
            output_path,
            stats,
            source_counts,
            deduplicate,
        )
        workbook.close()
        workbook = None
        temporary_path.replace(output_path)
    finally:
        if workbook is not None:
            workbook.close()
        if temporary_path.exists():
            temporary_path.unlink()

    return stats


def main() -> int:
    args = parse_args()
    source_path = args.pkl_path.expanduser()
    output_path = args.output.expanduser()

    try:
        print(f"正在加载索引：{source_path}")
        docstore, index_to_docstore_id = load_index(source_path)
        print(f"索引包含 {len(index_to_docstore_id):,} 条映射记录。")
        stats = export_to_excel(
            source_path,
            output_path,
            docstore,
            index_to_docstore_id,
            deduplicate=args.deduplicate,
            rows_per_sheet=args.rows_per_sheet,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, pickle.PickleError) as exc:
        print(f"导出失败：{exc}", file=sys.stderr)
        return 1

    print(
        f"导出完成：{stats.exported_rows:,} 条问答，"
        f"{stats.worksheet_count} 个问答工作表。"
    )
    if stats.skipped_empty_rows:
        print(f"跳过空问答：{stats.skipped_empty_rows:,} 条。")
    if stats.deduplicated_rows:
        print(f"去除完全重复问答：{stats.deduplicated_rows:,} 条。")
    if stats.truncated_cells:
        print(f"因 Excel 长度限制截断单元格：{stats.truncated_cells:,} 个。")
    print(f"Excel 文件：{output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
