from utils import determine_difficulty, Agent
from llm_client import create_llm_client
from md_agent import process_diff_query
from one_agent import process_base_query
from med_agent import process_mid_query
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import threading
from retriever import Retriever
import queue
import time
from typing import Generator, Dict, Any
from trace2skill_adapter.recorder import create_trace_recorder
from web_search import WebSearchError
from web_evidence import (
    build_source_lookup_answer,
    build_web_evidence_context,
    canonicalize_grounded_answer,
    format_evidence_summary,
    is_source_lookup_query,
    search_web_for_question,
)
from config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    DEFAULT_FAISS_VERSION,
    GENERATION_CONFIG_BASE,
    MAX_CONVERSATION_HISTORY,
    MAX_CONVERSATION_TURNS,
    MODEL_NAME,
    SERVER_HOST,
    SERVER_PORT,
    SSE_HEARTBEAT_SECONDS,
    STREAM_RETRIEVER_MIN_SCORE,
    TRACE2SKILL_CAPTURE_CONTENT,
    TRACE2SKILL_ENABLED,
    TRACE2SKILL_TRACE_DIR,
)

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

client = create_llm_client()
retrieverr = Retriever(DEFAULT_FAISS_VERSION, min_score=STREAM_RETRIEVER_MIN_SCORE)
# ✨ 新增：全局消息队列，用于收集中间过程
message_queues = {}
conversation_history = {}
job_states = {}
worker_threads = {}
job_state_lock = threading.Lock()
JOB_STALE_SECONDS = 180


def _start_job_state(session_id: str):
    now = time.time()
    with job_state_lock:
        job_states[session_id] = {
            "state": "running",
            "currentStage": "请求已接收，准备启动后台线程",
            "startedAt": now,
            "lastEventAt": now,
            "error": None,
        }


def _touch_job_state(
    session_id: str,
    *,
    state: str | None = None,
    current_stage: str | None = None,
    error: str | None = None,
):
    now = time.time()
    with job_state_lock:
        job = job_states.setdefault(
            session_id,
            {
                "state": "running",
                "currentStage": "后台任务运行中",
                "startedAt": now,
                "lastEventAt": now,
                "error": None,
            },
        )
        preserve_error = state == "completed" and job.get("state") == "error"
        if state is not None and not preserve_error:
            job["state"] = state
        if current_stage is not None and not preserve_error:
            job["currentStage"] = current_stage
        if error is not None:
            job["error"] = error
        job["lastEventAt"] = now


def get_job_status(session_id: str) -> Dict[str, Any]:
    now = time.time()
    with job_state_lock:
        job = dict(job_states.get(session_id) or {})
        worker = worker_threads.get(session_id)
    if not job:
        return {
            "sessionId": session_id,
            "state": "not_found",
            "workerAlive": False,
        }
    idle_seconds = max(0.0, now - job["lastEventAt"])
    elapsed_seconds = max(0.0, now - job["startedAt"])
    worker_alive = bool(worker and worker.is_alive())
    return {
        "sessionId": session_id,
        "state": job["state"],
        "currentStage": job["currentStage"],
        "workerAlive": worker_alive,
        "workerStoppedUnexpectedly": (
            job["state"] == "running" and not worker_alive
        ),
        "possiblyStalled": (
            job["state"] == "running"
            and worker_alive
            and idle_seconds >= JOB_STALE_SECONDS
        ),
        "idleSeconds": round(idle_seconds, 1),
        "elapsedSeconds": round(elapsed_seconds, 1),
        "lastEventAt": int(job["lastEventAt"] * 1000),
        "error": job.get("error"),
    }


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _optional_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_conversation_context(session_id: str, max_turns: int = MAX_CONVERSATION_TURNS) -> str:
    """
    获取对话上下文，只包含最近几轮的用户问题和最终答案
    Args:
        session_id: 会话ID
        max_turns: 最大保留的对话轮数
    Returns:
        格式化的对话上下文字符串
    """
    if session_id not in conversation_history:
        return ""
    
    history = conversation_history[session_id]
    # 只取最近的max_turns轮对话
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    if not recent_history:
        return ""
    
    context = "以下是之前的对话历史：\n"
    for i, turn in enumerate(recent_history, 1):
        context += f"用户问题: {turn['question']}\n"
        context += f"助手回答: {turn['answer']}\n"
    
    context += "\n请基于以上对话历史回答当前问题。\n\n"
    return context

def save_conversation_turn(session_id: str, question: str, answer: str):
    """
    保存一轮对话
    Args:
        session_id: 会话ID
        question: 用户问题
        answer: 最终答案
    """
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({
        "question": question,
        "answer": answer,
        "timestamp": int(time.time())
    })
    
    # 可选：限制历史记录长度，避免内存无限增长
    max_history_length = MAX_CONVERSATION_HISTORY
    if len(conversation_history[session_id]) > max_history_length:
        conversation_history[session_id] = conversation_history[session_id][-max_history_length:]

def create_question_with_context(question: str, session_id: str) -> str:
    """
    将当前问题与历史上下文结合
    Args:
        question: 当前用户问题
        session_id: 会话ID
    Returns:
        包含上下文的完整问题
    """
    context = get_conversation_context(session_id)
    if context:
        return f"{context}当前问题: {question}"
    else:
        return question
generation_config_base = GENERATION_CONFIG_BASE.copy()
generation_config_base["model"] = MODEL_NAME

# ✨ 新增：流式输出辅助类
class StreamHelper:
    @staticmethod
    def format_sse(data: Dict[str, Any]) -> str:
        """格式化SSE数据"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @staticmethod
    def send_step(session_id: str, agent_name: str, description: str, details: list = None):
        """发送执行步骤到队列"""
        _touch_job_state(
            session_id,
            state="running",
            current_stage=f"{agent_name}：{description}",
        )
        if session_id not in message_queues:
            return
            
        step_data = {
            "type": "agent_step",
            "step": {
                "agent": agent_name,
                "description": description,
                "details": details or [],
                "status": "processing",
                "timestamp": int(time.time() * 1000)
            }
        }
        message_queues[session_id].put(step_data)
    
    @staticmethod
    def send_output(session_id: str, agent_name: str, content: str, status: str = "completed"):
        """发送智能体输出到队列"""
        _touch_job_state(
            session_id,
            state="running",
            current_stage=f"{agent_name}：已产生新输出",
        )
        if session_id not in message_queues:
            return
            
        output_data = {
            "type": "agent_output",
            "output": {
                "agentName": agent_name,
                "content": content,
                "status": status,
                "isIncremental": False,
                "timestamp": int(time.time() * 1000)
            }
        }
        message_queues[session_id].put(output_data)
    
    @staticmethod
    def send_final(session_id: str, content: str, query: str):
        """发送最终结果"""
        _touch_job_state(
            session_id,
            state="finalizing",
            current_stage="最终结果已生成，准备结束流",
        )
        if session_id not in message_queues:
            return
            
        final_data = {
            "type": "final_result",
            "content": content,
            "originalQuery": query,
            "model": MODEL_NAME,
            "timestamp": int(time.time() * 1000)
        }
        message_queues[session_id].put(final_data)
        print(f"[SSE] final_result 已入队: session={session_id}")
    
    @staticmethod
    def send_complete(session_id: str, query: str):
        """发送完成信号"""
        _touch_job_state(
            session_id,
            state="completed",
            current_stage="处理完成",
        )
        if session_id not in message_queues:
            return
            
        complete_data = {
            "type": "complete",
            "originalQuery": query,
            "model": MODEL_NAME,
            "timestamp": int(time.time() * 1000)
        }
        message_queues[session_id].put(complete_data)
        print(f"[SSE] complete 已入队: session={session_id}")

    @staticmethod
    def send_error(session_id: str, error: str, query: str):
        """发送错误事件；调用方随后必须发送 complete，避免 SSE 无限等待。"""
        _touch_job_state(
            session_id,
            state="error",
            current_stage="处理失败",
            error=error,
        )
        if session_id not in message_queues:
            return

        error_data = {
            "type": "error",
            "error": error,
            "originalQuery": query,
            "model": MODEL_NAME,
            "timestamp": int(time.time() * 1000),
        }
        message_queues[session_id].put(error_data)
        print(f"[SSE] error 已入队: session={session_id}, error={error}")

# ✨ 新增：包装函数，在调用原函数前后添加流式输出
def create_callback(session_id: str, default_agent_name: str, trace_recorder=None):
    """为特定会话创建回调函数，支持动态智能体名称"""
    def callback(step_type: str, content: str, agent_name: str = None, details: list = None):
        """
        增强的回调函数参数：
        - step_type: 'step' 或 'output' 或 'incremental'
        - content: 步骤描述或输出内容
        - agent_name: 可选，指定智能体名称，如果不提供则使用默认名称
        - details: 可选的详细信息列表
        """
        # 如果没有指定 agent_name，使用默认名称
        current_agent_name = agent_name or default_agent_name
        if trace_recorder is not None:
            try:
                trace_recorder.record_event(
                    event_type=step_type,
                    content=content,
                    agent_name=current_agent_name,
                    details=details,
                )
            except Exception as trace_error:
                print(f"轨迹事件记录失败: {trace_error}")
        
        if step_type == 'step':
            StreamHelper.send_step(session_id, current_agent_name, content, details)
        elif step_type == 'output':
            StreamHelper.send_output(session_id, current_agent_name, content)
        elif step_type == 'incremental':
            # 增量输出
            output_data = {
                "type": "agent_output",
                "output": {
                    "agentName": current_agent_name,
                    "content": content,
                    "status": "processing",
                    "isIncremental": True,
                    "timestamp": int(time.time() * 1000)
                }
            }
            _touch_job_state(
                session_id,
                state="running",
                current_stage=f"{current_agent_name}：正在生成内容",
            )
            if session_id in message_queues:
                message_queues[session_id].put(output_data)
    
    return callback

# ✨ 修改包装函数，传入回调
def process_base_query_with_callback(
    question,
    session_id,
    llm_client=None,
    trace_recorder=None,
    web_evidence_context="",
):
    """使用回调函数版本的 process_base_query"""
    agent_name = "基础分析智能体"
    StreamHelper.send_step(session_id, "基础分析智能体", "处理简单问题")
    StreamHelper.send_step(session_id, "基础分析智能体", "生成回答", ["理解问题", "检索知识", "组织答案"])
    # 调用原函数，传入OpenAI客户端
    callback = create_callback(session_id, agent_name, trace_recorder=trace_recorder)
    active_client = llm_client or client
    result = process_base_query(
        question,
        active_client,
        retrieverr,
        callback=callback,
        web_evidence_context=web_evidence_context,
    )
    StreamHelper.send_output(session_id, "基础分析智能体", result)
    return result

def process_mid_query_with_callback(
    question,
    session_id,
    llm_client=None,
    trace_recorder=None,
    web_evidence_context="",
):
    """使用回调函数版本的 process_mid_query"""
    agent_name = "中等难度分析系统"
    callback = create_callback(session_id, agent_name, trace_recorder=trace_recorder)
    
    # 调用原函数，传入OpenAI客户端和回调函数
    active_client = llm_client or client
    final_decision, multiAgent = process_mid_query(
        question,
        active_client,
        callback=callback,
        web_evidence_context=web_evidence_context,
    )
    
    return final_decision, multiAgent

def process_diff_query_with_callback(
    question,
    session_id,
    llm_client=None,
    trace_recorder=None,
    web_evidence_context="",
):
    """使用回调函数版本的 process_diff_query"""
    agent_name = "高难度分析系统"
    callback = create_callback(session_id, agent_name, trace_recorder=trace_recorder)
    
    # 调用原函数，传入OpenAI客户端和回调函数
    active_client = llm_client or client
    final_decision, multiAgent = process_diff_query(
        question,
        active_client,
        callback=callback,
        trace_recorder=trace_recorder,
        web_evidence_context=web_evidence_context,
    )
    
    return final_decision, multiAgent

# 修改后台处理函数
def background_process(
    question: str,
    session_id: str,
    difficulty: str,
    enable_difficulty_agent: bool = False,
    test_mode: bool = False,
    enable_web_search: bool = False,
    web_search_query: str | None = None,
):
    """后台执行分析过程"""
    request_client = None
    trace_recorder = create_trace_recorder(
        enabled=TRACE2SKILL_ENABLED,
        output_dir=TRACE2SKILL_TRACE_DIR,
        session_id=session_id,
        question=question,
        model=MODEL_NAME,
        capture_content=TRACE2SKILL_CAPTURE_CONTENT,
    )
    try:
        # OpenAI/httpx 客户端在实际后台线程中创建，避免跨线程复用连接池。
        request_client = create_llm_client()
        if test_mode:
            question_with_context = question
        else:
            question_with_context = create_question_with_context(question, session_id)

        # 1. 难度评估只依据用户问题和对话，不让网页摘要改变路由。
        if enable_difficulty_agent:
            StreamHelper.send_step(
                session_id,
                "难度评估智能体",
                "正在判断问题难度",
                ["解析问题", "评估推理复杂度", "选择分析路径"],
            )
            difficulty = determine_difficulty(question_with_context)
            StreamHelper.send_output(
                session_id,
                "难度评估智能体",
                f"## 难度评估结果\n\n智能体判定当前问题为：**{difficulty}**",
            )
        elif difficulty in ["simple", "medium", "hard"]:
            difficulty_map = {"simple": "简单", "medium": "中等", "hard": "困难"}
            difficulty = difficulty_map[difficulty]
        else:
            # 兼容没有传入新开关的旧客户端。
            difficulty = "困难"

        # 2. 联网检索只接收当前问题，先规划检索式，再做来源与主题过滤。
        web_response = None
        web_search_error = None
        web_search_elapsed_seconds = None
        if enable_web_search:
            StreamHelper.send_step(
                session_id,
                "联网检索系统",
                "正在规划检索式并筛选公开网页资料",
                ["仅发送当前问题", "优先匹配指定官方域名", "过滤无关来源并编号"],
            )
            web_started_at = time.monotonic()
            try:
                web_response = search_web_for_question(
                    question,
                    override_query=web_search_query,
                )
                StreamHelper.send_output(
                    session_id,
                    "联网检索系统",
                    format_evidence_summary(web_response),
                )
            except WebSearchError as error:
                web_search_error = str(error)
                StreamHelper.send_output(
                    session_id,
                    "联网检索系统",
                    f"## 联网检索未生效\n\n{web_search_error}。已自动回退到本地知识流程。",
                )
            except Exception as error:
                web_search_error = "联网检索发生未预期错误"
                print(f"联网检索异常: {error}")
                StreamHelper.send_output(
                    session_id,
                    "联网检索系统",
                    f"## 联网检索未生效\n\n{web_search_error}。已自动回退到本地知识流程。",
                )
            finally:
                web_search_elapsed_seconds = round(
                    time.monotonic() - web_started_at,
                    3,
                )

        if trace_recorder is not None:
            trace_recorder.set_context(
                difficulty=difficulty,
                test_mode=test_mode,
                endpoint="/chat/stream",
                web_search_requested=enable_web_search,
                web_search_applied=web_response is not None,
                web_search_provider=web_response.provider if web_response else None,
                web_search_source_urls=(
                    [result.url for result in web_response.results]
                    if web_response
                    else []
                ),
                web_search_error=web_search_error,
                web_search_queries=list(web_response.queries) if web_response else [],
                web_search_rejected_count=(
                    web_response.rejected_count if web_response else 0
                ),
                web_search_elapsed_seconds=web_search_elapsed_seconds,
            )

        # 3. 资料来源查询走可核验快速路径；病例分析才进入医疗多智能体。
        multiAgent = ""
        web_evidence_context = (
            build_web_evidence_context(web_response) if web_response else ""
        )
        if web_response is not None and is_source_lookup_query(question):
            StreamHelper.send_step(
                session_id,
                "资料查询路由",
                "已识别为公开资料查询，跳过临床多智能体辩论",
                ["保留原始问题", "输出已核验来源", "避免无关专家扩写"],
            )
            final_decision = build_source_lookup_answer(question, web_response)
        elif difficulty == "简单":
            final_decision = process_base_query_with_callback(
                question_with_context,
                session_id,
                llm_client=request_client,
                trace_recorder=trace_recorder,
                web_evidence_context=web_evidence_context,
            )
        elif difficulty == "中等" or difficulty == "困难":
            final_decision, multiAgent = process_diff_query_with_callback(
                question_with_context,
                session_id,
                llm_client=request_client,
                trace_recorder=trace_recorder,
                web_evidence_context=web_evidence_context,
            )
        else:
            final_decision = "未知难度，无法处理"
        if web_response is not None:
            final_decision = canonicalize_grounded_answer(
                final_decision,
                web_response,
            )
        save_conversation_turn(session_id, question, final_decision)
        if trace_recorder is not None:
            try:
                trace_path = trace_recorder.finalize(
                    outcome="success",
                    final_answer=final_decision,
                )
                if trace_path is not None:
                    print(f"Trace2Skill 轨迹已写入: {trace_path}")
            except Exception as trace_error:
                print(f"Trace2Skill 轨迹落盘失败: {trace_error}")
        # 3. 发送最终结果和完成信号
        StreamHelper.send_final(session_id, final_decision, question)
        time.sleep(1)
        StreamHelper.send_complete(session_id, question)
        
    except Exception as e:
        error_message = f"处理过程中出现错误：{str(e)}"
        if trace_recorder is not None:
            try:
                trace_recorder.finalize(outcome="error", error=error_message)
            except Exception as trace_error:
                print(f"Trace2Skill 错误轨迹落盘失败: {trace_error}")
        StreamHelper.send_error(session_id, error_message, question)
        StreamHelper.send_complete(session_id, question)
    finally:
        if request_client is not None:
            request_client.close()


# ✨ 新增：流式接口
@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    question = data.get("query")
    session_id = data.get("id")
    print(session_id)
    enable_multi_agent = data.get("enableMultiAgent", False)
    difficulty = data.get("difficulty")
    enable_difficulty_agent = data.get("enableDifficultyAgent")
    enable_web_search = _as_bool(data.get("enableWebSearch"), False)
    web_search_query = _optional_text(data.get("webSearchQuery"))
    if enable_difficulty_agent is None:
        # 旧客户端未发送开关且未指定难度时，维持原来的自动判定行为。
        enable_difficulty_agent = not difficulty or difficulty == "auto"

    
    if not enable_multi_agent:
        # 如果不启用流式，返回普通响应
        return await chat_endpoint(request)
    
    # 清空并初始化消息队列
    _start_job_state(session_id)
    if session_id in message_queues:
        while not message_queues[session_id].empty():
            try:
                message_queues[session_id].get_nowait()
            except queue.Empty:
                break
    else:
        message_queues[session_id] = queue.Queue()
    
    # 启动后台处理线程
    thread = threading.Thread(
        target=background_process,
        kwargs={
            "question": question,
            "session_id": session_id,
            "difficulty": difficulty,
            "enable_difficulty_agent": _as_bool(enable_difficulty_agent),
            "enable_web_search": enable_web_search,
            "web_search_query": web_search_query,
        },
    )
    thread.daemon = True
    with job_state_lock:
        worker_threads[session_id] = thread
    thread.start()
    
    # 流式生成器
    async def stream_generator():
        last_emit_at = time.monotonic()
        try:
            while True:
                # 非阻塞获取消息
                if session_id in message_queues:
                    try:
                        message = message_queues[session_id].get_nowait()
                        yield StreamHelper.format_sse(message)
                        last_emit_at = time.monotonic()
                        # 如果是完成信号，结束流
                        if message.get("type") == "complete":
                            break
                            
                    except queue.Empty:
                        # 队列为空时定期发送 SSE 注释心跳，防止长推理期间
                        # WSL 转发层、代理或浏览器将空闲连接关闭。
                        if time.monotonic() - last_emit_at >= SSE_HEARTBEAT_SECONDS:
                            yield ": heartbeat\n\n"
                            yield StreamHelper.format_sse({
                                "type": "status",
                                "status": get_job_status(session_id),
                            })
                            last_emit_at = time.monotonic()
                        await asyncio.sleep(0.1)
                        continue
                else:
                    break
        except asyncio.CancelledError:
            print(f"[SSE] 客户端连接已取消: session={session_id}")
            raise
        except Exception as e:
            print(f"[SSE] 流式传输异常: session={session_id}, error={e}")
            error_data = {
                "type": "error",
                "error": f"流式传输错误: {str(e)}"
            }
            yield StreamHelper.format_sse(error_data)
        finally:
            # 正常完成、处理失败或客户端中断都清理队列，避免会话资源泄漏。
            message_queues.pop(session_id, None)
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/chat/status/{session_id}")
async def chat_status_endpoint(session_id: str):
    """查询后台线程、当前阶段和最近一次进展，辅助区分长耗时与卡死。"""
    return get_job_status(session_id)


# ✨ 保留：原有的非流式接口
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("query")
    session_id = data.get("id")
    enable_web_search = _as_bool(data.get("enableWebSearch"), False)
    web_search_query = _optional_text(data.get("webSearchQuery"))
    trace_recorder = create_trace_recorder(
        enabled=TRACE2SKILL_ENABLED,
        output_dir=TRACE2SKILL_TRACE_DIR,
        session_id=session_id,
        question=question,
        model=MODEL_NAME,
        capture_content=TRACE2SKILL_CAPTURE_CONTENT,
    )
    
    # 原有逻辑保持不变
    if session_id in message_queues:
        while not message_queues[session_id].empty():
            try:
                message_queues[session_id].get_nowait()
            except queue.Empty:
                break
    else:
        message_queues[session_id] = queue.Queue()

    if session_id == '40101':
        difficulty = "简单"
    elif session_id == '40102':
        difficulty = "中等"
    elif session_id == '40103':
        difficulty = "困难"
    else:
        difficulty = determine_difficulty(question)

    web_response = None
    web_search_error = None
    web_search_elapsed_seconds = None
    if enable_web_search:
        web_started_at = time.monotonic()
        try:
            web_response = search_web_for_question(
                question,
                override_query=web_search_query,
            )
        except WebSearchError as error:
            web_search_error = str(error)
        except Exception as error:
            web_search_error = "联网检索发生未预期错误"
            print(f"联网检索异常: {error}")
        finally:
            web_search_elapsed_seconds = round(
                time.monotonic() - web_started_at,
                3,
            )

    if trace_recorder is not None:
        trace_recorder.set_context(
            difficulty=difficulty,
            endpoint="/chat",
            web_search_requested=enable_web_search,
            web_search_applied=web_response is not None,
            web_search_provider=web_response.provider if web_response else None,
            web_search_source_urls=(
                [result.url for result in web_response.results]
                if web_response
                else []
            ),
            web_search_error=web_search_error,
            web_search_queries=list(web_response.queries) if web_response else [],
            web_search_rejected_count=(
                web_response.rejected_count if web_response else 0
            ),
            web_search_elapsed_seconds=web_search_elapsed_seconds,
        )
    
    multiAgent = ""
    parsedSchedule = ""
    web_evidence_context = (
        build_web_evidence_context(web_response) if web_response else ""
    )
    
    if web_response is not None and is_source_lookup_query(question):
        final_decision = build_source_lookup_answer(question, web_response)
    elif difficulty == "简单":
        final_decision = process_base_query(
            question,
            client,
            web_evidence_context=web_evidence_context,
        )
        print(final_decision)
    elif difficulty == "中等":
        final_decision, multiAgent = process_mid_query(
            question,
            client,
            web_evidence_context=web_evidence_context,
        )
        print(final_decision)
    elif difficulty == "困难":
        final_decision, multiAgent = process_diff_query(
            question,
            client,
            trace_recorder=trace_recorder,
            web_evidence_context=web_evidence_context,
        )
        print(multiAgent)
    if web_response is not None:
        final_decision = canonicalize_grounded_answer(
            final_decision,
            web_response,
        )
    if trace_recorder is not None:
        try:
            trace_path = trace_recorder.finalize(
                outcome="success",
                final_answer=final_decision,
            )
            if trace_path is not None:
                print(f"Trace2Skill 轨迹已写入: {trace_path}")
        except Exception as trace_error:
            print(f"Trace2Skill 轨迹落盘失败: {trace_error}")
    
    return {
        "最终结果": final_decision,
        "难度": difficulty,
        "多智能体结果": multiAgent,
        "多智能体调度结果": parsedSchedule,
        "联网检索": {
            "requested": enable_web_search,
            "applied": web_response is not None,
            "provider": web_response.provider if web_response else None,
            "sources": (
                [result.url for result in web_response.results]
                if web_response
                else []
            ),
            "error": web_search_error,
            "queries": list(web_response.queries) if web_response else [],
            "rejectedCount": web_response.rejected_count if web_response else 0,
            "elapsedSeconds": web_search_elapsed_seconds,
        },
        "id": session_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
