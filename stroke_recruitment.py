"""Hard-constrained expert recruitment for stroke-related questions.

The language model may recommend expert IDs, but it cannot create roles, rename
experts, edit their descriptions, or replace their system prompts.  Final routing
is always validated against the authoritative registry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from config import STROKE_EXPERT_REGISTRY_PATH


CORE_EXPERT_ID = "stroke_neurology"
DEFAULT_FILL_ORDER = (
    "neuroimaging",
    "emergency_stroke",
    "clinical_pharmacy_coagulation",
    "cardioembolic_stroke",
    "neurocritical_care",
    "neurosurgery",
    "neurointervention",
    "stroke_rehabilitation",
    "neurocognitive_psychiatry",
    "pediatric_stroke",
    "pregnancy_stroke",
)


@dataclass(frozen=True)
class ExpertDefinition:
    id: str
    name: str
    description: str
    system_prompt: str


class ExpertRegistry:
    """Validated, immutable view of the configured stroke expert whitelist."""

    def __init__(self, experts: Sequence[ExpertDefinition]):
        self.experts = tuple(experts)
        self.by_id = {expert.id: expert for expert in self.experts}
        self.by_name = {expert.name: expert for expert in self.experts}
        if len(self.experts) != 12:
            raise ValueError(f"卒中专家库必须恰好包含 12 名专家，当前为 {len(self.experts)} 名。")
        if len(self.by_id) != len(self.experts):
            raise ValueError("卒中专家库包含重复 ID。")
        if len(self.by_name) != len(self.experts):
            raise ValueError("卒中专家库包含重复名称。")
        if CORE_EXPERT_ID not in self.by_id:
            raise ValueError("卒中专家库缺少核心专家 stroke_neurology。")
        for expert in self.experts:
            if not all((expert.id, expert.name, expert.description, expert.system_prompt)):
                raise ValueError(f"专家 {expert.id or '<unknown>'} 的固定字段不完整。")


@dataclass(frozen=True)
class RecruitmentDecision:
    is_stroke: bool
    experts: tuple[ExpertDefinition, ...]
    scenario_tags: tuple[str, ...]
    rule_id: str
    source: str
    raw_recommendation: str
    repairs: tuple[str, ...]

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return tuple(expert.id for expert in self.experts)

    def to_legacy_text(self) -> str:
        return "\n".join(
            f"{index}. {expert.name} - {expert.description} - 层级结构：独立"
            for index, expert in enumerate(self.experts, start=1)
        )

    def to_agents_data(self) -> list[tuple[str, str]]:
        return [
            (f"{index}. {expert.name} - {expert.description}", "独立")
            for index, expert in enumerate(self.experts, start=1)
        ]

    def audit_details(self) -> list[str]:
        details = [
            f"路由来源：{self.source}",
            f"规则：{self.rule_id}",
            f"最终专家 ID：{', '.join(self.expert_ids)}",
        ]
        details.extend(f"修正：{repair}" for repair in self.repairs)
        return details


def load_expert_registry(path: str | Path = STROKE_EXPERT_REGISTRY_PATH) -> ExpertRegistry:
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法加载卒中专家库 {registry_path}: {exc}") from exc

    raw_experts = payload.get("experts") if isinstance(payload, dict) else None
    if not isinstance(raw_experts, list):
        raise ValueError("卒中专家库缺少 experts 数组。")
    try:
        experts = [
            ExpertDefinition(
                id=str(item["id"]).strip(),
                name=str(item["name"]).strip(),
                description=str(item["description"]).strip(),
                system_prompt=str(item["system_prompt"]).strip(),
            )
            for item in raw_experts
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"卒中专家库记录格式无效: {exc}") from exc
    return ExpertRegistry(experts)


def is_stroke_related(question: str) -> bool:
    """Conservative gate that keeps unrelated medical questions on the old path."""

    text = (question or "").lower()
    explicit_terms = (
        "卒中", "脑梗", "脑出血", "短暂性脑缺血", "tia", "蛛网膜下腔出血",
        "脑静脉系统血栓", "脑静脉窦血栓", "取栓", "溶栓", "再灌注",
        "大血管闭塞", "大脑中动脉", "基底动脉闭塞", "颈内动脉",
        "脑动静脉畸形", "颅内动脉瘤",
    )
    if any(term in text for term in explicit_terms):
        return True

    sudden_terms = ("突发", "突然", "急性", "醒来发现")
    focal_terms = (
        "偏瘫", "无力", "单侧无力", "一侧肢体无力", "失语", "口齿不清",
        "言语不清", "面歪", "复视", "四肢无力",
    )
    return any(term in text for term in sudden_terms) and any(
        term in text for term in focal_terms
    )


def detect_scenario_tags(question: str) -> tuple[str, ...]:
    text = (question or "").lower()
    tag_terms = (
        ("pediatric", ("儿童", "患儿", "小儿", "岁儿童")),
        ("pregnancy_postpartum", ("孕", "妊娠", "产后", "产褥")),
        ("hemorrhage", ("脑出血", "蛛网膜下腔出血", "血肿")),
        ("large_vessel_occlusion", ("大血管闭塞", "大脑中动脉", "基底动脉闭塞", "颈内动脉末端", "取栓")),
        ("anticoagulation", ("抗凝", "阿哌沙班", "达比加群", "利伐沙班", "华法林", "机械瓣")),
        ("rehabilitation", ("康复", "吞咽", "呛咳", "日常生活")),
        ("cognition_mood", ("认知", "抑郁", "自伤", "活着没有意义", "答非所问")),
        ("emergency", ("突发", "突然", "意识下降", "血氧下降", "转院", "转运")),
    )
    return tuple(tag for tag, terms in tag_terms if any(term in text for term in terms))


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def hard_route_for_question(question: str) -> tuple[str, tuple[str, str, str]] | None:
    """Return deterministic routes for safety-critical and validated scenarios."""

    text = (question or "").lower()
    core = CORE_EXPERT_ID

    pediatric = _has_any(text, ("儿童", "患儿", "小儿", "岁儿童"))
    pregnancy = _has_any(text, ("孕", "妊娠", "产后", "产褥"))
    lvo = _has_any(
        text,
        ("大血管闭塞", "大脑中动脉", "基底动脉闭塞", "颈内动脉末端", "取栓"),
    )
    if pediatric and lvo:
        return "pediatric_lvo", (core, "pediatric_stroke", "neurointervention")
    if pediatric:
        return "pediatric_stroke", (core, "pediatric_stroke", "neuroimaging")
    if pregnancy:
        return "pregnancy_or_postpartum", (core, "pregnancy_stroke", "neuroimaging")

    aspiration_red_flags = _has_any(text, ("呛咳", "误吸")) and _has_any(
        text, ("发热", "呼吸急促", "血氧下降", "低氧")
    )
    if aspiration_red_flags:
        return "home_aspiration_red_flags", (core, "stroke_rehabilitation", "emergency_stroke")

    if _has_any(text, ("低血糖", "补糖")):
        return "stroke_mimic_with_residual_deficit", (core, "emergency_stroke", "neuroimaging")

    transfer_lvo = lvo and _has_any(
        text, ("基层医院", "不能取栓", "转运目的地", "减少延误")
    )
    if transfer_lvo:
        return "lvo_transfer", (core, "emergency_stroke", "neurointervention")

    anticoagulants = _has_any(
        text, ("阿哌沙班", "达比加群", "利伐沙班", "华法林", "抗凝", "xa 因子"),
    )
    hemorrhage = _has_any(text, ("脑出血", "蛛网膜下腔出血", "血肿"))
    if hemorrhage and anticoagulants:
        return "anticoagulant_related_ich", (core, "neurocritical_care", "clinical_pharmacy_coagulation")

    surgical_hemorrhage = hemorrhage and _has_any(
        text,
        (
            "意识恶化", "嗜睡", "意识模糊", "意识评分下降", "意识下降",
            "小脑出血", "第四脑室", "脑室扩大", "脑积水", "脑室引流",
            "脑干受压", "神经外科", "反复呕吐",
        ),
    )
    if surgical_hemorrhage:
        return "ich_neurocritical_neurosurgical", (core, "neurocritical_care", "neurosurgery")

    vascular_lesion = _has_any(
        text,
        ("动脉瘤", "蛛网膜下腔出血", "动静脉畸形", "血管畸形", "夹闭", "介入栓塞"),
    )
    if vascular_lesion:
        return "aneurysm_or_avm", (core, "neurointervention", "neurosurgery")

    if lvo:
        return "large_vessel_occlusion", (core, "neuroimaging", "neurointervention")

    thrombolysis_anticoagulation = anticoagulants and _has_any(
        text, ("溶栓", "急性脑梗", "突发偏瘫", "疑似急性")
    )
    if thrombolysis_anticoagulation:
        return "thrombolysis_with_anticoagulant", (core, "neuroimaging", "clinical_pharmacy_coagulation")

    if _has_any(text, ("颈动脉", "颈内动脉起始段", "血运重建")):
        return "symptomatic_carotid_disease", (core, "neuroimaging", "neurointervention")

    if _has_any(text, ("卵圆孔未闭", "pfo", "隐源性")):
        return "cryptogenic_or_pfo", (core, "cardioembolic_stroke", "neuroimaging")

    cardio_pharmacy = _has_any(
        text,
        ("房颤", "机械瓣", "人工心脏瓣膜", "人工机械瓣", "他汀不耐受", "多种他汀", "血脂"),
    ) and _has_any(text, ("抗凝", "栓塞", "二级预防", "他汀", "血脂", "恢复抗凝"))
    if cardio_pharmacy:
        return "cardioembolic_secondary_prevention", (core, "cardioembolic_stroke", "clinical_pharmacy_coagulation")

    rehabilitation = _has_any(text, ("康复", "吞咽", "失语", "日常生活"))
    cognitive_or_mood = _has_any(
        text, ("认知", "抑郁", "自伤", "活着没有意义", "答非所问", "记不住")
    )
    stable_followup = _has_any(text, ("两周", "一个月", "三个月", "生命体征稳定", "卒中后"))
    if rehabilitation and (cognitive_or_mood or stable_followup):
        return "rehabilitation_cognition_mood", (core, "stroke_rehabilitation", "neurocognitive_psychiatry")

    acute_focal = _has_any(text, ("突发", "突然", "醒来发现", "急性")) and _has_any(
        text, ("偏瘫", "无力", "失语", "口齿不清", "精细活动障碍")
    )
    uncertain_onset_or_imaging = _has_any(
        text,
        ("尚未做", "尚无影像", "只有平扫", "最后正常时间不清", "睡前正常", "ct 报告未见出血", "低 nihss"),
    )
    if acute_focal or uncertain_onset_or_imaging:
        return "acute_stroke_triage", (core, "emergency_stroke", "neuroimaging")

    return None


def build_stroke_recruiter_prompt(registry: ExpertRegistry) -> str:
    roster = "\n".join(
        f"- {expert.id}: {expert.name}（{expert.description}）"
        for expert in registry.experts
    )
    return (
        "你是卒中专家推荐器。只能从下列白名单推荐专家 ID；不得创建、改名或改写专家。\n"
        "卒中神经内科专家必须入选，总数必须恰好为 3，ID 不得重复。\n"
        "只输出 JSON 对象，格式为："
        '{"scenario_tags":["标签"],"expert_ids":["stroke_neurology","...","..."]}。\n'
        f"专家白名单：\n{roster}"
    )


def build_stroke_recruitment_request(question: str) -> str:
    return f"请为以下问题推荐 3 名专家，仅返回约定的 JSON：\n问题：{question}"


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _extract_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_model_recommendation(
    raw_recommendation: str,
    registry: ExpertRegistry,
) -> tuple[list[str], list[str]]:
    """Parse JSON first, then conservatively recover known IDs/names from text."""

    raw = (raw_recommendation or "").strip()
    payload = _extract_first_json_object(raw)
    candidate_values: list[str] = []
    tags: list[str] = []
    if payload:
        raw_ids = payload.get("expert_ids", [])
        if isinstance(raw_ids, list):
            candidate_values.extend(str(value).strip() for value in raw_ids)
        raw_tags = payload.get("scenario_tags", [])
        if isinstance(raw_tags, list):
            tags = [str(value).strip() for value in raw_tags if str(value).strip()]

    if not candidate_values:
        positions: list[tuple[int, str]] = []
        for expert in registry.experts:
            for token in (expert.id, expert.name):
                position = raw.find(token)
                if position >= 0:
                    positions.append((position, expert.id))
        candidate_values = [expert_id for _, expert_id in sorted(positions)]

    normalized: list[str] = []
    for value in candidate_values:
        if value in registry.by_id:
            normalized.append(value)
        elif value in registry.by_name:
            normalized.append(registry.by_name[value].id)
    return _deduplicate(normalized), _deduplicate(tags)


def resolve_stroke_recruitment(
    question: str,
    raw_recommendation: str,
    registry: ExpertRegistry,
    num_agents: int = 3,
) -> RecruitmentDecision:
    if num_agents != 3:
        raise ValueError("卒中硬约束路由当前要求恰好招募 3 名专家。")
    if not is_stroke_related(question):
        raise ValueError("非卒中问题不应进入卒中硬约束路由。")

    parsed_ids, model_tags = parse_model_recommendation(raw_recommendation, registry)
    repairs: list[str] = []
    hard_route = hard_route_for_question(question)
    if hard_route:
        rule_id, final_ids = hard_route
        if tuple(parsed_ids) != final_ids:
            repairs.append("模型推荐已按确定性场景规则覆盖")
        source = "hard_rule"
    else:
        rule_id = "validated_model_or_fallback"
        valid_ids = list(parsed_ids)
        if CORE_EXPERT_ID not in valid_ids:
            valid_ids.insert(0, CORE_EXPERT_ID)
            repairs.append("补入必选核心专家")
        if len(valid_ids) > num_agents:
            valid_ids = [CORE_EXPERT_ID] + [
                value for value in valid_ids if value != CORE_EXPERT_ID
            ][:num_agents - 1]
            repairs.append("裁剪为 3 名白名单专家")
        for fallback_id in DEFAULT_FILL_ORDER:
            if len(valid_ids) >= num_agents:
                break
            if fallback_id not in valid_ids:
                valid_ids.append(fallback_id)
                repairs.append(f"回退补入 {fallback_id}")
        final_ids = tuple(valid_ids[:num_agents])
        source = "validated_model" if parsed_ids else "deterministic_fallback"

    experts = tuple(registry.by_id[expert_id] for expert_id in final_ids)
    tags = tuple(_deduplicate([*detect_scenario_tags(question), *model_tags]))
    return RecruitmentDecision(
        is_stroke=True,
        experts=experts,
        scenario_tags=tags,
        rule_id=rule_id,
        source=source,
        raw_recommendation=raw_recommendation or "",
        repairs=tuple(repairs),
    )


def build_fixed_expert_system_prompt(expert: ExpertDefinition) -> str:
    """Return only the immutable registry prompt for downstream skill injection."""

    return expert.system_prompt
