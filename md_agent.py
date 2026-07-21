import json
import random
import time
import concurrent.futures
from threading import Lock
from typing import Any, Dict, List
from tqdm import tqdm
from termcolor import cprint
from pptree import Node
from pptree import *
from prettytable import PrettyTable 
from utils import Agent
from retriever import Retriever
from config import (
    DEFAULT_FAISS_VERSION,
    MEDICAL_SKILL_DIR,
    MEDICAL_SKILL_ENABLED,
    MEDICAL_SKILL_MAX_CHARS,
    STROKE_HARD_RECRUITMENT_ENABLED,
    STROKE_EXPERT_REGISTRY_PATH,
)
from stroke_recruitment import (
    build_fixed_expert_system_prompt,
    build_stroke_recruiter_prompt,
    build_stroke_recruitment_request,
    is_stroke_related,
    load_expert_registry,
    resolve_stroke_recruitment,
)
from trace2skill_adapter.skill_loader import inject_skill, load_medical_skill

# 线程安全锁
callback_lock = Lock()
result_lock = Lock()
interaction_lock = Lock()
retriever = Retriever(DEFAULT_FAISS_VERSION)
def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    print(info)
    count = 0
    for expert, hierarchy in info:
        
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = '独立'
        print(hierarchy)
        if '>'  in hierarchy:
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()
            print(1)
            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1
    print("fin")
    return agents


def _extract_json_array(text: str) -> List[Any]:
    """从主智能体回复中提取首个 JSON 数组。"""
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            result = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return []

    return result if isinstance(result, list) else []


def _resolve_agent_name(requested_name: str, valid_agent_names: List[str]) -> str:
    requested = (requested_name or "").strip().lower()
    if not requested:
        return ""

    normalized_names = {name.strip().lower(): name for name in valid_agent_names}
    if requested in normalized_names:
        return normalized_names[requested]

    for normalized, original in normalized_names.items():
        if requested in normalized or normalized in requested:
            return original
    return ""


def create_subquestion_plan(
    question: str,
    agent_profiles: List[Dict[str, str]],
    master_agent: Agent,
    max_subquestions: int = 5,
) -> List[Dict[str, str]]:
    """由主智能体统一拆题，并把每个子问题分配给已招募的从智能体。"""
    valid_agent_names = [profile["name"] for profile in agent_profiles]
    roster = "\n".join(
        f"- {profile['name']}：{profile['description']}"
        for profile in agent_profiles
    )
    prompt = f"""
你是医疗多智能体系统中的主智能体。请把原始问题拆分为 2 至 {max_subquestions} 个
可独立回答、互补且不重复的医学子问题，并把每个子问题匹配给最合适的一个从智能体。

原始问题：
{question}

可用从智能体：
{roster}

要求：
1. 子问题合起来必须覆盖原始问题的关键诊断、检查、治疗、风险或随访维度；
2. 不得虚构原始问题中不存在的患者事实；
3. assigned_agent 必须严格使用上面列出的从智能体名称；
4. 同一从智能体可以处理多个子问题；
5. 只输出 JSON 数组，不要输出解释或 Markdown。

输出格式：
[
  {{
    "sub_question": "可独立回答的子问题",
    "assigned_agent": "从智能体名称",
    "why_needed": "拆分及分配理由"
  }}
]
""".strip()

    response = master_agent.chat(prompt)
    raw_plan = _extract_json_array(response)
    plan = []

    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        sub_question = (item.get("sub_question") or "").strip()
        assigned_agent = _resolve_agent_name(
            item.get("assigned_agent") or "",
            valid_agent_names,
        )
        if not sub_question or not assigned_agent:
            continue
        if any(task["sub_question"].lower() == sub_question.lower() for task in plan):
            continue
        plan.append({
            "task_id": f"SQ{len(plan) + 1}",
            "sub_question": sub_question,
            "assigned_agent": assigned_agent,
            "why_needed": (item.get("why_needed") or "").strip(),
        })
        if len(plan) == max_subquestions:
            break

    minimum_tasks = min(2, max_subquestions, len(agent_profiles))
    if plan and len(plan) < minimum_tasks:
        assigned_names = {task["assigned_agent"] for task in plan}
        for profile in agent_profiles:
            if profile["name"] in assigned_names:
                continue
            plan.append({
                "task_id": f"SQ{len(plan) + 1}",
                "sub_question": (
                    f"从{profile['name']}的专业角度，原始问题中还需要核实哪些关键医学条件、"
                    f"风险或限制：{question}"
                ),
                "assigned_agent": profile["name"],
                "why_needed": "补足主智能体规划，使复杂问题至少包含两个互补任务。",
            })
            assigned_names.add(profile["name"])
            if len(plan) >= minimum_tasks:
                break

    if plan:
        return plan

    # 主智能体输出异常时保持系统可用，并确保每个专家仍获得一个明确任务。
    fallback_plan = []
    for index, profile in enumerate(agent_profiles[:max_subquestions]):
        fallback_plan.append({
            "task_id": f"SQ{index + 1}",
            "sub_question": (
                f"请从{profile['name']}的专业角度，分析原始问题中与该领域相关的"
                f"诊断、治疗、风险及限制条件：{question}"
            ),
            "assigned_agent": profile["name"],
            "why_needed": "主智能体规划结果无法解析，使用按专家角色生成的保底任务。",
        })
    return fallback_plan


def format_task_plan(task_plan: List[Dict[str, str]]) -> str:
    lines = ["## 主智能体子问题规划", ""]
    for task in task_plan:
        lines.extend([
            f"### {task['task_id']} → {task['assigned_agent']}",
            f"- 子问题：{task['sub_question']}",
            f"- 分配理由：{task['why_needed'] or '未提供'}",
            "",
        ])
    return "\n".join(lines).strip()


def collect_assigned_tasks_concurrent(
    agent_name,
    agent,
    question,
    assignments,
    client,
    callback=None,
    agent_index=0,
    total_agents=0,
):
    """检索指定子问题，并将结构化任务包发送给对应从智能体。"""
    if callback:
        with callback_lock:
            callback(
                'step',
                f'执行已分配子问题 ({agent_index + 1}/{total_agents})',
                agent_name=agent_name,
                details=[f"任务数：{len(assignments)}", "每题生成 3 个检索改写", "合并去重知识"],
            )

    task_records = []
    for task in assignments:
        retrieval_result = retriever.retrieve_docs_for_subquestion(
            sub_question=task["sub_question"],
            model=client,
            rewrite_count=3,
        )
        task_record = dict(task)
        task_record.update(retrieval_result)
        task_records.append(task_record)

    packet_sections = []
    for record in task_records:
        rewrites_text = "\n".join(
            f"  {index}. {rewrite}"
            for index, rewrite in enumerate(record["rewrites"], start=1)
        )
        knowledge_text = "\n".join(
            f"  [{index}] {knowledge}"
            for index, knowledge in enumerate(record["knowledge"], start=1)
        ) or "  未检索到达到阈值的知识，请明确说明证据不足并基于专业知识谨慎回答。"
        packet_sections.append(f"""
### {record['task_id']}
子问题：{record['sub_question']}
任务理由：{record['why_needed'] or '未提供'}
检索改写：
{rewrites_text}
检索知识：
{knowledge_text}
""".strip())

    task_packets = "\n\n".join(packet_sections)
    print(agent_name, task_packets)
    try:
        opinion = agent.chat(f"""
你是医疗多智能体系统中的从智能体。主智能体已经把原始问题拆分并向你分配了以下任务。
请只负责你收到的子问题；结合检索知识和你的专业知识逐题回答，不要改写成对完整原问题的泛泛回答。

原始问题（仅用于理解上下文）：
{question}

你的结构化任务包：
{task_packets}

请严格按以下结构回复每个任务：
### 任务编号
子问题结论：
关键依据：
风险与不确定性：
给主智能体的摘要：
""".strip(), callback=callback, agent_name=agent_name)
        
        if callback:
            with callback_lock:
                callback('output', f'## 从智能体任务回复\n\n{opinion}', agent_name=agent_name)
        
        return agent_name, opinion, task_records
    except Exception as e:
        error_msg = f"专家 {agent_name} 分析过程中出现错误: {str(e)}"
        print(error_msg)
        if callback:
            with callback_lock:
                callback('output', f'## {agent_name} 专家分析遇到问题\n\n{error_msg}', agent_name=agent_name)
        return agent_name, error_msg, task_records

def collect_final_answer_concurrent(
    agent_index,
    agent,
    question,
    shared_context="",
    callback=None,
    round_num=0,
    assigned_tasks=None,
):
    """并发收集单个专家的最终答案"""
    agent_name = agent.role
    assigned_tasks = assigned_tasks or []
    assigned_tasks_text = "\n".join(
        f"- {task['task_id']}：{task['sub_question']}"
        for task in assigned_tasks
    ) or "- 本专家未被主智能体直接分配子问题，请作为交叉评审专家审查共享上下文。"
    
    try:
        response = agent.chat(f"""既然您已经与其他医学专家进行了互动，请结合以下共享上下文，回顾您的专业知识和本轮讨论中其他专家的评论，并提交给主智能体的当前最终回复。

原始问题：
{question}

主智能体最初分配给您的子问题：
{assigned_tasks_text}

本轮共享上下文：
{shared_context}

请明确说明：
1. 对每个已分配子问题的当前结论；
2. 您同意或不同意哪些专家观点；
3. 经过本轮讨论后，您的观点是否发生变化；
4. 您仍然存在的疑问或风险。

答案：""",callback=callback, agent_name=agent_name)
        
        if callback:
            with callback_lock:
                callback('output', f'## 轮次 {round_num} 最终观点\n\n{response}', agent_name=agent_name)
        
        return agent_name, response
    except Exception as e:
        error_msg = f"专家 {agent_name} 最终答案收集出现错误: {str(e)}"
        print(error_msg)
        if callback:
            with callback_lock:
                callback('output', f'## {agent_name} 最终答案收集遇到问题\n\n{error_msg}', agent_name=agent_name)
        return agent_name, error_msg

def check_participation_concurrent(agent_index, agent, context, callback=None):
    """并发检查专家是否参与讨论"""
    agent_name = agent.role
    agent_id_str = f"代理 {agent_index+1}"
    
    try:
        participate = agent.chat(f"""根据您团队中其他医学专家的意见（如下所示）:

{context}

现在是交流阶段，我建议你与其他专家进行讨论，可以向他们问问题或者反驳他们的观点或对于不确定的部分进行讨论，输出Yes进行交流，如果你百分百确认自己正确且没有任何交流的必要，则输出No不进行交流。请说明您是否想与专家交谈,请只回答"Yes"或"No",不要输出任何其他额外内容""")
        # """根据您团队中其他医学专家的意见（如下所示）:

# {context}

# 有任何你没把握的地方都推荐你进行交流讨论，请说明您是否想与任何专家交谈,请只回答"Yes"或"No",不要输出任何其他额外内容"""
        print(participate)
        wants_to_participate = 'yes' in participate.lower().strip()
        return agent_index, agent_id_str, agent_name, wants_to_participate, participate
    except Exception as e:
        error_msg = f"检查专家 {agent_name} 参与意愿时出错: {str(e)}"
        print(error_msg)
        return agent_index, agent_id_str, agent_name, False, error_msg

def run_concurrent_assigned_tasks(
    agent_dict,
    question,
    task_plan,
    client,
    callback=None,
    max_workers=None,
):
    """按主智能体的任务规划，并发执行各从智能体的子问题。"""
    assignments_by_agent = {agent_name: [] for agent_name in agent_dict}
    for task in task_plan:
        agent_name = task.get("assigned_agent")
        if agent_name in assignments_by_agent:
            assignments_by_agent[agent_name].append(task)

    active_assignments = [
        (agent_name, agent_dict[agent_name], assignments)
        for agent_name, assignments in assignments_by_agent.items()
        if assignments
    ]
    if not active_assignments:
        return {}, "", []

    if max_workers is None:
        max_workers = min(len(active_assignments), 4)

    opinions = {}
    report_parts = []
    task_execution_log = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, (agent_name, agent, assignments) in enumerate(active_assignments):
            future = executor.submit(
                collect_assigned_tasks_concurrent,
                agent_name,
                agent,
                question,
                assignments,
                client,
                callback,
                index,
                len(active_assignments),
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                agent_name, opinion, task_records = future.result()
                with result_lock:
                    opinions[agent_name.lower()] = opinion
                    report_parts.append(f"({agent_name.lower()}): {opinion}")
                    task_execution_log.extend(task_records)
            except Exception as exc:
                print(f'从智能体子问题执行产生异常: {exc}')

    task_order = {task["task_id"]: index for index, task in enumerate(task_plan)}
    task_execution_log.sort(
        key=lambda record: task_order.get(record.get("task_id"), len(task_order))
    )
    return opinions, "\n".join(report_parts), task_execution_log

def run_concurrent_final_answers(
    medical_agents,
    question,
    shared_context="",
    callback=None,
    round_num=0,
    task_plan=None,
    max_workers=None,
):
    """并发收集所有专家的最终答案"""
    if max_workers is None:
        max_workers = min(len(medical_agents), 4)
    
    final_answers = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, agent in enumerate(medical_agents):
            assigned_tasks = [
                task for task in (task_plan or [])
                if task.get("assigned_agent") == agent.role
            ]
            future = executor.submit(
                collect_final_answer_concurrent,
                i,
                agent,
                question,
                shared_context,
                callback,
                round_num,
                assigned_tasks,
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                agent_name, response = future.result()
                with result_lock:
                    final_answers[agent_name] = response
                print(f"    代理 ({agent_name}) 的答案已收集。")
            except Exception as exc:
                print(f'专家最终答案收集产生异常: {exc}')
    
    return final_answers

def run_concurrent_participation_check(medical_agents, context, callback=None, max_workers=None):
    """并发检查所有专家的参与意愿"""
    if max_workers is None:
        max_workers = min(len(medical_agents), 4)
    
    participation_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i, agent in enumerate(medical_agents):
            future = executor.submit(
                check_participation_concurrent,
                i, agent, context, callback
            )
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                with result_lock:
                    participation_results.append(result)
            except Exception as exc:
                print(f'专家参与检查产生异常: {exc}')
    
    # 按索引排序，保持原始顺序
    participation_results.sort(key=lambda x: x[0])
    return participation_results

def process_diff_query(
    question,
    client,
    callback=None,
    need_rag=False,
    trace_recorder=None,
):
    """
    处理复杂度的医疗查询，适用于中文场景。
    callback: 可选的回调函数 callback(step_type, content, agent_name=None, details=None)
    """
    
    # if need_rag:
    #     rag_knowledge_raw = retriever.retrieve_docs_multi_channel(question=question, model=client)
    #     main_docs = rag_knowledge_raw["main_docs"]
    #     sub_docs = rag_knowledge_raw["sub_docs"]
    #     rag_knowledge = []
    #     if main_docs is not None:
    #         rag_knowledge = rag_knowledge + [data['a'] for data in main_docs]
    #     if sub_docs is not None:
    #         rag_knowledge = rag_knowledge + [data['a'] for data in sub_docs]
    #     question=question + "参考信息为：" + '\t'.join(rag_knowledge)

    active_skill = None
    if MEDICAL_SKILL_ENABLED:
        active_skill = load_medical_skill(
            MEDICAL_SKILL_DIR,
            max_chars=MEDICAL_SKILL_MAX_CHARS,
        )
    if trace_recorder is not None:
        trace_recorder.set_context(
            medical_skill_enabled=active_skill is not None,
            medical_skill_name=active_skill.name if active_skill else None,
            medical_skill_version=active_skill.version if active_skill else None,
        )

    # 第1步：专家招募
    cprint("[信息] 第1步：专家招募", 'yellow', attrs=['blink'])
    
    use_fixed_stroke_recruitment = (
        STROKE_HARD_RECRUITMENT_ENABLED and is_stroke_related(question)
    )
    stroke_registry = None
    stroke_decision = None
    fixed_expert_by_name = {}
    if use_fixed_stroke_recruitment:
        stroke_registry = load_expert_registry(STROKE_EXPERT_REGISTRY_PATH)
        recruit_prompt = inject_skill(
            build_stroke_recruiter_prompt(stroke_registry),
            role="招募者",
            skill=active_skill,
        )
    else:
        recruit_prompt = inject_skill(
            "您是一位经验丰富的医学专家，负责招募一组具有不同身份背景的专家，"
            "并要求他们讨论并解决给定的医疗问题。",
            role="招募者",
            skill=active_skill,
        )
    tmp_agent = Agent(
        client,
        role_message=recruit_prompt,
        role='招募者',
        trace_recorder=trace_recorder,
    )
    
    num_agents = 3
    
    if callback:
        callback('step', f'招募 {num_agents} 名医学专家', agent_name='专家招募系统')

    if use_fixed_stroke_recruitment:
        recruitment_request = build_stroke_recruitment_request(question)
    else:
        recruitment_request = f"""问题：{question}\n
您需要招募 {num_agents} 名具有不同医学专业知识的专家。考虑到医疗问题所涉及到的不同专业知识，您会招募哪些专家以便更好地做出准确的回答？
此外，您还需要指定专家之间的沟通结构，或者说明他们是独立工作的。

例如,如果你要招募五名专家，他们的沟通结构为：呼吸科医生 == 新生儿科医生 == 医学遗传学家 == 儿科医生 > 心脏病医生，你的输出应该如下：
1. 儿科医生 - 专注于婴幼儿、儿童和青少年的医疗保健。 - 层级结构：独立
2. 心脏病医生 - 专注于心血管相关疾病的诊断和治疗。 - 层级结构：儿科医生 > 心脏病医生
3. 呼吸科医生 - 专注于呼吸系统疾病的诊断和治疗。 - 层级结构：独立
4. 新生儿科医生 - 专注于新生儿的护理，特别是早产儿或出生时有医疗问题的新生儿。 - 层级结构：独立
5. 医学遗传学家 - 专注于基因和遗传的研究。 - 层级结构：独立

请严格按照上述格式回答，不要包含任何您的理由及其他多余信息。"""

    raw_recruited = tmp_agent.chat(
        recruitment_request,
        callback=None if use_fixed_stroke_recruitment else callback,
        agent_name='专家招募系统',
    )

    if use_fixed_stroke_recruitment:
        stroke_decision = resolve_stroke_recruitment(
            question=question,
            raw_recommendation=raw_recruited or "",
            registry=stroke_registry,
            num_agents=num_agents,
        )
        recruited = stroke_decision.to_legacy_text()
        fixed_expert_by_name = {
            expert.name: expert for expert in stroke_decision.experts
        }
        if trace_recorder is not None:
            trace_recorder.set_context(
                stroke_hard_recruitment=True,
                stroke_recruitment_source=stroke_decision.source,
                stroke_recruitment_rule=stroke_decision.rule_id,
                stroke_expert_ids=list(stroke_decision.expert_ids),
                stroke_recruitment_repairs=list(stroke_decision.repairs),
            )
        if callback:
            callback(
                'step',
                '校验并固化卒中专家团队',
                agent_name='专家招募系统',
                details=stroke_decision.audit_details(),
            )
    else:
        recruited = raw_recruited
        if not recruited or not recruited.strip():
            raise RuntimeError("专家招募模型未返回任何内容，复杂问题处理已终止。")
    
    cprint("招募信息", 'yellow', attrs=['blink'])
    print(recruited)
    
    if callback:
        callback('output', f'## 专家招募结果\n\n{recruited}', agent_name='专家招募系统')
    
    # 解析招募信息。卒中路径直接使用校验后的注册表记录，避免二次文本漂移。
    if stroke_decision is not None:
        agents_data = stroke_decision.to_agents_data()
    else:
        agents_info = [agent_info.split(" - 层级结构：") for agent_info in recruited.split('\n') if '- 层级结构：' in agent_info]
        agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]
    if not agents_data:
        raise RuntimeError(
            "专家招募结果格式无效：没有解析到“专家 - 描述 - 层级结构”记录。"
        )

    # Agent Emojis
    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    # 解析层级结构
    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    # 创建代理列表字符串
    agent_list = ""
    for i, agent in enumerate(agents_data):
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
            agent_list += f"代理 {i+1}: {agent_role} - {description}\n"
        except:
            agent_role = agent[0].split('-')[0].strip().lower()
            description = agent[0].split('-')[1].strip().lower() if '-' in agent[0] else "（无详细描述）"
            agent_list += f"代理 {i+1}: {agent_role} - {description}\n"

    if callback:
        callback('step', '初始化专家团队', agent_name='团队管理系统')

    # 初始化Agent实例
    agent_dict = {}
    medical_agents = []
    agent_profiles = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except IndexError:
            agent_role = agent[0].split('-')[0].strip().lower()
            description = agent[0].split('-')[1].strip().lower() if '-' in agent[0] else "（无详细描述）"
        except Exception as e:
            print(f"解析代理信息时出错: {agent[0]}, 错误: {e}")
            continue

        fixed_expert = fixed_expert_by_name.get(agent_role)
        if fixed_expert is not None:
            agent_role = fixed_expert.name
            description = fixed_expert.description
            base_inst_prompt = build_fixed_expert_system_prompt(fixed_expert)
        else:
            base_inst_prompt = (
                f"您是一名 {agent_role}领域专家，专长是 {description}。"
                "您的工作是与团队中的其他医学专家协作。"
            )
        inst_prompt = inject_skill(
            base_inst_prompt,
            role=agent_role,
            skill=active_skill,
        )
        _agent = Agent(
            client,
            role_message=inst_prompt,
            role=agent_role,
            trace_recorder=trace_recorder,
        )
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)
        agent_profiles.append({
            "name": agent_role,
            "description": description,
        })

    # 生成专家团队总结
    agent_summary = "## 专家团队组建完成\n\n"
    for idx, agent in enumerate(agents_data):
        try:
            agent_info = f"**专家 {idx+1}** ({agent_emoji[idx]}): {agent[0].split('-')[0].strip()}\n"
            agent_summary += agent_info
            print(f"代理 {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except IndexError:
            agent_info = f"**专家 {idx+1}** ({agent_emoji[idx]}): {agent[0].strip()}\n"
            agent_summary += agent_info
            print(f"代理 {idx+1} ({agent_emoji[idx]}): {agent[0].strip()}")
        except Exception as e:
            print(f"打印代理信息时出错: {agent[0]}, 错误: {e}")

    if callback:
        callback('output', agent_summary, agent_name='团队管理系统')

    print()
    # 第2步：协作决策制定
    if callback:
        callback('step', '开始协作决策制定', agent_name='协作系统')
    
    cprint("[信息] 第2步：协作决策制定", 'yellow', attrs=['blink'])
    cprint("[信息] 第2.1步：层级结构选择", 'yellow', attrs=['blink'])
    try:
        print_tree(hierarchy_agents[0], horizontal=False)
    except IndexError:
        print("[警告] 未能生成层级结构树（可能没有招募到代理或解析失败）。")
    print()

    # 设置交互轮次和回合数
    num_rounds = 2
    num_turns = 2
    num_agents = len(medical_agents)
    if num_agents == 0:
        error_message = "未成功初始化任何医学从智能体，无法进行协作。"
        print(f"[错误] {error_message}")
        if callback:
            callback('output', f"## 错误\n\n{error_message}", agent_name='系统错误')
        raise RuntimeError(error_message)

    # 主智能体负责统一拆题、任务匹配、阶段总结、一致性判断和最终回答。
    master_agent_prompt = inject_skill(
        (
            "您是医疗多智能体系统的主智能体。您负责将原始问题拆分并分配给合适的"
            "从智能体，审查各从智能体的证据和回复，识别一致意见、冲突与证据缺口，"
            "最后生成谨慎、完整且可追溯的综合回答。"
        ),
        role="主智能体",
        skill=active_skill,
    )
    master_agent = Agent(
        client,
        role_message=master_agent_prompt,
        role='主智能体',
        trace_recorder=trace_recorder,
    )

    if callback:
        callback(
            'step',
            '拆分问题并匹配从智能体',
            agent_name='主智能体',
            details=['生成互补子问题', '匹配专业专家', '校验任务覆盖范围'],
        )
    task_plan = create_subquestion_plan(
        question=question,
        agent_profiles=agent_profiles,
        master_agent=master_agent,
        max_subquestions=5,
    )
    task_plan_text = format_task_plan(task_plan)
    print(task_plan_text)
    if callback:
        callback('output', task_plan_text, agent_name='主智能体')

    # 初始化交互日志
    interaction_log = {f'轮次 {round_num}': {f'回合 {turn_num}': {f'代理 {source_agent_num}': {f'代理 {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}
    print_log = {}
    
    # 第2.2步：按照主智能体规划并发执行子问题
    if callback:
        callback('step', '派发子问题任务', agent_name='主智能体')
    
    cprint("[信息] 第2.2步：参与式辩论", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    
    print("[信息] 按任务规划获取从智能体回复...")
    
    # 【并发优化1】：按专家分组，并发执行主智能体分配的子问题
    if callback:
        callback('step', '并发执行从智能体任务', agent_name='并发协调器')
    
    round_opinions[1], initial_report, task_execution_log = run_concurrent_assigned_tasks(
        agent_dict=agent_dict,
        question=question,
        task_plan=task_plan,
        client=client,
        callback=callback,
    )
    
    print(initial_report)
    print_log["任务规划"] = task_plan
    print_log["任务执行"] = task_execution_log
    print_log["初步意见"] = initial_report
    
    final_answer = None
    print_log["交流过程"] = ""
    
    # 开始多轮辩论
    for n in range(1, num_rounds+1):
        print(f"== 轮次 {n} ==")
        round_name = f"轮次 {n}"
        
        if callback:
            callback('step', f'第 {n} 轮专家辩论', agent_name='辩论协调器')

        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        # 回复回传主智能体，由主智能体执行阶段总结和一致性判断。
        report = master_agent.chat(f'''以下是从智能体针对已分配子问题提交的报告。\n\n{assessment}\n\n您需要完成以下步骤：
1. 仔细全面地考虑所有报告。
2. 按子问题检查报告是否回答了对应任务，并提取关键知识和证据。
3. 识别专家之间的一致意见、直接冲突和证据不足之处。
4. 指出下一轮需要继续讨论或交叉验证的问题。
5. 基于这些内容给出阶段性综合分析，但不要掩盖不确定性。\n
您应该严格按照以下格式输出：
子问题完成情况：[逐项说明各子问题是否得到回答]
关键知识：[您的关键知识总结]
一致意见：[专家共同认可的内容]
争议焦点：[尚未解决的分歧]
证据缺口：[检索知识或专家回答中仍缺少的证据]
待讨论问题：[下一轮需要讨论的问题]
总体分析：[您的阶段性综合分析]''')
        print(f"  轮次 {n} 总结报告已生成。")

        base_round_context = f"""原始专家观点：
{assessment}

主智能体阶段性总结：
{report}""".strip()

        if callback:
            callback('incremental', f'## 轮次 {n} 阶段性总结\n\n{report}', agent_name='主智能体')

        num_yes_total_round = 0
        round_interactions = ""
        
        # 执行多回合交互
        for turn_num in range(num_turns):
            turn_name = f"回合 {turn_num + 1}"
            print(f"  |_{turn_name}")

            num_yes_turn = 0
            current_turn_interactions = []

            # 收集先前评论作为上下文
            all_prior_comments = ""
            if n > 1 or turn_num > 0:
                for r in range(1, n + 1):
                    for t in range(1, (turn_num + 1 if r == n else num_turns + 1)):
                        round_key = f"轮次 {r}"
                        turn_key = f"回合 {t}"
                        if round_key in interaction_log and turn_key in interaction_log[round_key]:
                            for source_agent_idx, targets in interaction_log[round_key][turn_key].items():
                                for target_agent_idx, comment in targets.items():
                                    if comment:
                                        all_prior_comments += f"{source_agent_idx} -> {target_agent_idx}: {comment}\n"

            context_for_participation_prompt = base_round_context
            if all_prior_comments:
                context_for_participation_prompt += f"""

此前专家交互记录：
{all_prior_comments}"""

            # 【并发优化2】：并发检查每个代理是否参与讨论
            if callback:
                callback('step', f'并发检查专家参与意愿', agent_name='参与协调器')
            
            participation_results = run_concurrent_participation_check(
                medical_agents, context_for_participation_prompt, callback
            )

            # 处理参与结果和后续交互
            for agent_index, agent_id_str, agent_name, wants_to_participate, participate_response in participation_results:
                agent_v = medical_agents[agent_index]
                
                print(participate_response)
                if wants_to_participate:
                    num_yes_turn += 1
                    num_yes_total_round += 1

                    chosen_expert_str = agent_v.chat(f"""请输入您想与之交谈的专家编号（只输入数字，多个用逗号隔开）：
{agent_list}
例如，如果您想与 代理 1. 儿科医生 交谈，请只返回 1。如果您想与多位专家交谈，请返回 1,2""")

                    try:
                        chosen_experts_indices = [int(ce.strip()) for ce in chosen_expert_str.replace('，', ',').split(',') if ce.strip().isdigit()]
                    except ValueError:
                        print(f"  [警告] {agent_id_str} 返回了无效的专家编号: '{chosen_expert_str}'，跳过此回合的发言。")
                        continue
                    
                    for ce_idx in chosen_experts_indices:
                        if 1 <= ce_idx <= len(medical_agents):
                            target_agent_id_str = f"代理 {ce_idx}"
                            specific_question = agent_v.chat(f"""请首先简单重申您的医学专业领域，然后向您选择的专家 ({target_agent_id_str}. {medical_agents[ce_idx-1].role}) 提出您的意见或问题,或者回答他之前提出的问题。请在有足够把握时，以简洁的理由清晰表达，力求说服对方。""")
                            
                            interaction_text = f"    {agent_id_str} ({agent_emoji[agent_index]} {medical_agents[agent_index].role}) -> {target_agent_id_str} ({agent_emoji[ce_idx-1]} {medical_agents[ce_idx-1].role}) : {specific_question}"
                            print(interaction_text)
                            round_interactions += f"\n{interaction_text}"
                            print_log["交流过程"] += f"\n{interaction_text}"
                            
                            # 线程安全地记录交互
                            with interaction_lock:
                                if round_name not in interaction_log: 
                                    interaction_log[round_name] = {}
                                if turn_name not in interaction_log[round_name]: 
                                    interaction_log[round_name][turn_name] = {f'代理 {i+1}': {} for i in range(num_agents)}
                                if agent_id_str not in interaction_log[round_name][turn_name]: 
                                    interaction_log[round_name][turn_name][agent_id_str] = {}

                                interaction_log[round_name][turn_name][agent_id_str][target_agent_id_str] = specific_question
                                current_turn_interactions.append(f"{agent_id_str} -> {target_agent_id_str}: {specific_question}")
                        else:
                            print(f"  [警告] {agent_id_str} 选择了无效的专家编号: {ce_idx}，跳过。")
                else:
                    silence_text = f"    {agent_id_str} ({agent_emoji[agent_index]} {agent_v.role}): \U0001f910 (选择不发言)"
                    print(silence_text)
                    round_interactions += f"\n{silence_text}"
                    print_log["交流过程"] += f"\n{silence_text}"

            if num_yes_turn == 0:
                print(f"  回合 {turn_num + 1} 中无代理发言，结束此轮次。")
                break

        # 发送本轮交互情况
        if callback and round_interactions:
            callback('incremental', f'### 轮次 {n} 专家交互记录\n{round_interactions}\n\n', agent_name='交互记录器')

        # 检查是否有代理在本轮发言
        if num_yes_total_round == 0 and n > 1:
            print(f"轮次 {n} 中无代理进行有效讨论，提前结束辩论。")
            if callback:
                callback('step', f'轮次 {n} 无有效讨论，提前结束', agent_name='辩论协调器')
            break

        # 【并发优化3】：并发收集本轮最终答案
        if callback:
            callback('step', f'并发收集轮次 {n} 最终答案', agent_name='答案收集器')
        
        print(f"  轮次 {n} 结束，收集中间答案...")
        
        round_shared_context = f"""{base_round_context}

本轮专家交互记录：
{round_interactions if round_interactions else "本轮暂无额外专家交互。"}""".strip()

        tmp_round_final_answer = run_concurrent_final_answers(
            medical_agents=medical_agents,
            question=question,
            shared_context=round_shared_context,
            callback=callback,
            round_num=n,
            task_plan=task_plan,
        )

        round_answers[round_name] = tmp_round_final_answer
        final_answer = tmp_round_final_answer

        # 本轮最终观点作为下一轮的初始专家意见，使下一轮总结和讨论
        # 能显式读取所有专家在上一轮形成的最新结论。
        if n < num_rounds:
            round_opinions[n + 1] = dict(tmp_round_final_answer)

    # 第3步：最终决策
    if callback:
        callback('step', '开始最终决策阶段', agent_name='最终决策系统')

    # 生成交互日志表格
    print('\n交互日志摘要')
    myTable = PrettyTable([''] + [f"代理 {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"代理 {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('---')
            else:
                agent_i_str = f"代理 {i}"
                agent_j_str = f"代理 {j}"
                i2j = False
                j2i = False
                # 检查所有轮次和回合的交互
                for r_idx in range(1, num_rounds + 1):
                    r_key = f"轮次 {r_idx}"
                    if r_key in interaction_log:
                        for t_idx in range(1, num_turns + 1):
                            t_key = f"回合 {t_idx}"
                            if t_key in interaction_log[r_key]:
                                # 检查 i -> j
                                if agent_i_str in interaction_log[r_key][t_key] and \
                                   agent_j_str in interaction_log[r_key][t_key][agent_i_str] and \
                                   interaction_log[r_key][t_key][agent_i_str][agent_j_str]:
                                    i2j = True
                                # 检查 j -> i
                                if agent_j_str in interaction_log[r_key][t_key] and \
                                   agent_i_str in interaction_log[r_key][t_key][agent_j_str] and \
                                   interaction_log[r_key][t_key][agent_j_str][agent_i_str]:
                                    j2i = True
                            if i2j and j2i: 
                                break
                        if i2j and j2i: 
                            break

                # 根据交互情况添加符号
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u2709 ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u2709 ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u21F5 ({i}<->{j})')

        myTable.add_row(row)

    print(myTable)

    cprint("\n[信息] 第3步：最终决策", 'yellow', attrs=['blink'])

    if callback:
        callback(
            'step',
            '主智能体执行最终总结与一致性判断',
            agent_name='主智能体',
            details=['核对子问题覆盖', '识别一致与冲突', '形成最终回答'],
        )

    # 准备最终答案字符串
    final_answer_str = ""
    final_ans = {}
    if final_answer:
        final_answer_str = "\n".join([f"专家 {role}: {ans}" for role, ans in final_answer.items()])
        final_ans = {role: ans for role, ans in final_answer.items()}
    else:
        final_answer_str = initial_report
        print("[警告] 未能从辩论中获取最终答案，将使用初步意见进行最终决策。")

    # 由同一个主智能体接收从智能体回复并生成最终总结。
    final_decision_content = master_agent.chat(f"""请根据主智能体最初的子问题规划和各从智能体的最终答案（或初步意见），完成最终一致性审查并回答原始问题。

子问题规划：
{task_plan_text}

各从智能体的意见：
{final_answer_str}

原始问题：
{question}

要求：
1. 检查所有子问题是否得到覆盖；
2. 明确列出专家一致意见；
3. 明确列出专家冲突、证据不足和仍然存在的不确定性；
4. 冲突不能仅按多数票忽略，应根据证据质量、专业匹配度和风险进行判断；
5. 最终回答必须直接回应原始问题，并保留必要的医学安全提醒。

请严格遵循以下格式：
子问题结论：[逐项汇总]
一致性判断：[一致意见]
冲突与不确定性：[分歧、证据缺口及处理方式]
最终回答：[对用户原始问题的综合回答]
""", callback=callback, agent_name='主智能体')
    
    print("主智能体的最终决策:", final_decision_content)

    if callback:
        callback('output', f'## 主智能体最终医疗决策\n\n{final_decision_content}', agent_name='主智能体')

    if callback:
        callback('step', '复杂分析完成', agent_name='系统总结')

    return final_decision_content, final_ans
