"""Question-aware web evidence planning, filtering, and grounding.

This module deliberately sits above ``web_search``. The transport module is
responsible for talking to a provider; this module decides what should be
searched, which results are admissible, and how evidence is exposed to agents.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence
from urllib.parse import urlparse

from web_search import SearchResult, WebSearchError, search_web


@dataclass(frozen=True)
class SearchPlan:
    original_query: str
    queries: tuple[str, ...]
    required_domains: tuple[str, ...] = ()
    topic_terms: tuple[str, ...] = ()
    source_lookup: bool = False


@dataclass(frozen=True)
class EvidenceSearchResponse:
    provider: str
    results: list[SearchResult]
    queries: tuple[str, ...]
    rejected_count: int = 0
    required_domains: tuple[str, ...] = ()


_SOURCE_TERMS = re.compile(
    r"资料|来源|链接|出处|官网|查询|检索|文献|指南|publication|source|reference|link",
    re.IGNORECASE,
)
_CASE_TERMS = re.compile(
    r"患者|病人|症状|体征|诊断|鉴别|用药|剂量|处方|病例|检查结果|"
    r"patient|symptom|diagnos|treat|dose|case",
    re.IGNORECASE,
)
_WHO_PATTERN = re.compile(r"(?<![A-Za-z])WHO(?![A-Za-z])|世界卫生组织", re.IGNORECASE)
_WHOIS_PATTERN = re.compile(
    r"\bwhois\b|whoisxmlapi|godaddy|dynadot|域名注册|注册信息",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
)
_MARKDOWN_LINK_START_RE = re.compile(r"\[([^\]\n]+)\]\(")
_TRAILING_URL_PUNCTUATION = ".,;，。"

_TOPIC_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"脑卒中|卒中|中风|\bstroke\b", re.IGNORECASE), ("stroke", "卒中")),
    (
        re.compile(r"心肌梗死|心梗|myocardial infarction", re.IGNORECASE),
        ("myocardial infarction", "heart attack", "心肌梗死"),
    ),
    (re.compile(r"糖尿病|diabetes", re.IGNORECASE), ("diabetes", "糖尿病")),
    (re.compile(r"高血压|hypertension", re.IGNORECASE), ("hypertension", "高血压")),
    (re.compile(r"癌症|肿瘤|cancer|tumou?r", re.IGNORECASE), ("cancer", "癌症", "肿瘤")),
)


def _normalise_question(question: str) -> str:
    return " ".join((question or "").split()).strip()


def _topic_terms(question: str) -> tuple[str, ...]:
    terms: list[str] = []
    for pattern, candidates in _TOPIC_RULES:
        if pattern.search(question):
            terms.extend(candidates)
    return tuple(dict.fromkeys(terms))


def _english_topic_terms(terms: Sequence[str]) -> list[str]:
    return [term for term in terms if term.isascii()]


def is_source_lookup_query(question: str) -> bool:
    """Return true for public-source lookup requests rather than patient cases."""

    normalised = _normalise_question(question)
    if not normalised or _CASE_TERMS.search(normalised):
        return False
    return bool(_SOURCE_TERMS.search(normalised))


def plan_web_search(question: str, override_query: str | None = None) -> SearchPlan:
    """Build provider queries and explicit authority constraints."""

    normalised = _normalise_question(question)
    if not normalised:
        raise WebSearchError("联网检索问题不能为空")

    required_domains: tuple[str, ...] = ()
    if _WHO_PATTERN.search(normalised) and not _WHOIS_PATTERN.search(normalised):
        required_domains = ("who.int",)

    topics = _topic_terms(normalised)
    source_lookup = is_source_lookup_query(normalised)
    override = _normalise_question(override_query or "")
    if override:
        queries = (override,)
    elif required_domains == ("who.int",):
        english_topics = _english_topic_terms(topics) or ["health"]
        subject = " ".join(english_topics)
        qualifiers: list[str] = []
        if re.search(r"指南|guideline", normalised, re.IGNORECASE):
            qualifiers.append("guideline")
        if re.search(r"资料|事实|fact", normalised, re.IGNORECASE):
            qualifiers.append("fact sheet")
        suffix = " ".join(dict.fromkeys(qualifiers))
        queries = (f"site:who.int {subject} {suffix}".strip(),)
    else:
        queries = (normalised[:400],)

    return SearchPlan(
        original_query=normalised,
        queries=queries,
        required_domains=required_domains,
        topic_terms=topics,
        source_lookup=source_lookup,
    )


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _domain_allowed(hostname: str, required_domains: Sequence[str]) -> bool:
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in required_domains
    )


def _is_relevant(item: SearchResult, plan: SearchPlan) -> bool:
    host = _hostname(item.url)
    if not host:
        return False
    combined = f"{item.title} {item.snippet} {item.url} {host}"
    asks_for_whois = bool(_WHOIS_PATTERN.search(plan.original_query))
    if not asks_for_whois and _WHOIS_PATTERN.search(combined):
        return False
    if plan.required_domains and not _domain_allowed(host, plan.required_domains):
        return False
    if plan.topic_terms:
        lowered = combined.casefold()
        if not any(term.casefold() in lowered for term in plan.topic_terms):
            return False
    return True


def _deduplicate(items: Iterable[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in items:
        key = item.url.strip().rstrip("/").casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def search_web_for_question(
    question: str,
    override_query: str | None = None,
) -> EvidenceSearchResponse:
    """Search using a plan, then reject irrelevant or non-authoritative hits."""

    plan = plan_web_search(question, override_query)
    accepted: list[SearchResult] = []
    rejected_count = 0
    provider = ""
    for query in plan.queries:
        raw = search_web(query)
        provider = raw.provider
        for item in raw.results:
            if _is_relevant(item, plan):
                accepted.append(item)
            else:
                rejected_count += 1

    accepted = _deduplicate(accepted)
    if not accepted:
        constraint = (
            f"（要求来源域名：{', '.join(plan.required_domains)}）"
            if plan.required_domains
            else ""
        )
        raise WebSearchError(
            "联网服务有响应，但没有返回与当前问题匹配的可信证据"
            f"{constraint}；已拒绝 {rejected_count} 条结果"
        )
    return EvidenceSearchResponse(
        provider=provider,
        results=accepted,
        queries=plan.queries,
        rejected_count=rejected_count,
        required_domains=plan.required_domains,
    )


def build_web_evidence_context(response: EvidenceSearchResponse) -> str:
    """Render an immutable evidence ledger with stable W1/W2 identifiers."""

    lines = ["联网证据账本（仅以下 URL 可作为本次联网引用；引用时必须保留证据编号）："]
    for index, item in enumerate(response.results, start=1):
        lines.extend(
            [
                f"[W{index}] 标题：{item.title}",
                f"[W{index}] URL：{item.url}",
                f"[W{index}] 摘要：{item.snippet or '搜索服务未提供摘要'}",
            ]
        )
    return "\n".join(lines)


def format_evidence_summary(response: EvidenceSearchResponse) -> str:
    lines = [
        f"实际检索式：{'；'.join(response.queries)}",
        (
            f"服务：{response.provider}；采纳 {len(response.results)} 条，"
            f"拒绝 {response.rejected_count} 条不相关结果。"
        ),
    ]
    for index, item in enumerate(response.results, start=1):
        lines.append(f"[W{index}] {item.title}\n{item.url}")
    return "\n".join(lines)


def build_source_lookup_answer(
    question: str,
    response: EvidenceSearchResponse,
) -> str:
    """Return a deterministic answer for a source-only lookup request."""

    authority = (
        f"并按指定官方域名（{', '.join(response.required_domains)}）筛选"
        if response.required_domains
        else "并进行了相关性筛选"
    )
    lines = [
        f"已联网查询“{_normalise_question(question)}”，{authority}。",
        "",
        "可核验来源：",
    ]
    for index, item in enumerate(response.results, start=1):
        lines.append(f"- [W{index}] [{item.title}]({item.url})")
        if item.snippet:
            lines.append(f"  {item.snippet}")
    lines.extend(
        [
            "",
            (
                f"检索服务：{response.provider}；采纳 {len(response.results)} 条，"
                f"拒绝 {response.rejected_count} 条。"
            ),
            "以上为搜索服务返回的公开网页摘要；重要医学结论请打开原始页面核验。",
        ]
    )
    return "\n".join(lines)


def _approved_urls(response: EvidenceSearchResponse) -> set[str]:
    return {item.url.rstrip("/") for item in response.results}


def _split_url_candidate(candidate: str) -> tuple[str, str]:
    """Separate trailing prose/Markdown punctuation from a balanced URL."""

    url = candidate
    suffix = ""
    while url and url[-1] in _TRAILING_URL_PUNCTUATION:
        suffix = url[-1] + suffix
        url = url[:-1]
    while url.endswith(")") and url.count(")") > url.count("("):
        suffix = ")" + suffix
        url = url[:-1]
    return url, suffix


def _extract_urls(text: str) -> set[str]:
    urls = set()
    for match in _URL_RE.finditer(text or ""):
        url, _ = _split_url_candidate(match.group(0))
        if url:
            urls.add(url.rstrip("/"))
    return urls


def _replace_markdown_links(text: str, approved: set[str]) -> str:
    """Validate Markdown links using balanced parentheses in the target URL."""

    output = []
    cursor = 0
    while True:
        match = _MARKDOWN_LINK_START_RE.search(text, cursor)
        if match is None:
            output.append(text[cursor:])
            break

        output.append(text[cursor:match.start()])
        url_start = match.end()
        depth = 1
        index = url_start
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1

        if depth != 0:
            output.append(text[match.start():])
            break

        label = match.group(1)
        url = text[url_start:index].strip()
        if url.startswith(("http://", "https://")) and url.rstrip("/") not in approved:
            output.append(f"{label}（未验证链接已移除）")
        else:
            output.append(text[match.start():index + 1])
        cursor = index + 1
    return "".join(output)


def grounding_issues(answer: str, response: EvidenceSearchResponse) -> list[str]:
    """Return machine-checkable web-grounding failures."""

    approved = _approved_urls(response)
    found = _extract_urls(answer or "")
    unsupported = sorted(found - approved)
    issues: list[str] = []
    if not found.intersection(approved):
        issues.append("回答没有引用任何已采纳的联网来源")
    if unsupported:
        issues.append(f"回答包含未获准 URL：{', '.join(unsupported)}")
    if response.results and re.search(
        r"未(?:能)?(?:检索|搜索|查找|找到).{0,12}(?:资料|来源|结果|信息)",
        answer or "",
    ):
        issues.append("已有联网证据，但回答仍声称未找到资料")
    return issues


def canonicalize_grounded_answer(
    answer: str,
    response: EvidenceSearchResponse,
) -> str:
    """Remove invented URLs and append canonical evidence when it is missing."""

    approved = _approved_urls(response)

    cleaned = _replace_markdown_links(answer or "", approved)

    def replace_bare(match: re.Match[str]) -> str:
        url, suffix = _split_url_candidate(match.group(0))
        if url.rstrip("/") in approved:
            return url + suffix
        return "（未验证链接已移除）" + suffix

    cleaned = _URL_RE.sub(replace_bare, cleaned)
    if response.results:
        cleaned = re.sub(
            r"未(?:能)?(?:检索|搜索|查找|找到).{0,12}(?:资料|来源|结果|信息)",
            "已获得可核验的联网来源（见下方来源列表）",
            cleaned,
        )
    if not any(url in cleaned for url in approved):
        source_lines = ["", "---", "联网来源（系统校验）："]
        for index, item in enumerate(response.results, start=1):
            source_lines.append(f"- [W{index}] [{item.title}]({item.url})")
        cleaned = cleaned.rstrip() + "\n" + "\n".join(source_lines)
    return cleaned
