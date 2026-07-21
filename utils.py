import json
import re
import time
from config import (
    GENERATION_CONFIG_BASE,
    GENERATION_CONFIG_GREEDY,
    LLM_ENABLE_THINKING,
    MODEL_NAME,
    SERVE_URL,
    TEMP_RESPONSE_MAX_TOKENS,
    TEMP_RESPONSE_STREAM,
    TEMP_RESPONSE_TEMPERATURES,
)
from llm_client import create_llm_client

_default_client = None


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)


def apply_thinking_instruction(message: str) -> str:
    """Use Qwen3's documented soft switch unless thinking was explicitly enabled."""
    if LLM_ENABLE_THINKING or "/no_think" in message:
        return message
    return f"{message.rstrip()}\n\n/no_think"


def strip_thinking_content(content: str) -> str:
    """Remove Qwen-style thinking blocks before parsing or displaying content."""
    if not content:
        return ""
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_RE.sub("", content)
    return content.strip()


class ThinkingTagFilter:
    """Incrementally remove <think> blocks even when tags span stream chunks."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buffer = ""
        self._inside_thinking = False

    def feed(self, content: str) -> str:
        if not content:
            return ""
        self._buffer += content
        output = []

        while self._buffer:
            lowered = self._buffer.lower()
            if self._inside_thinking:
                close_index = lowered.find(self._CLOSE)
                if close_index == -1:
                    keep = min(len(self._buffer), len(self._CLOSE) - 1)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[close_index + len(self._CLOSE):]
                self._inside_thinking = False
                continue

            open_index = lowered.find(self._OPEN)
            if open_index != -1:
                output.append(self._buffer[:open_index])
                self._buffer = self._buffer[open_index + len(self._OPEN):]
                self._inside_thinking = True
                continue

            keep = min(len(self._buffer), len(self._OPEN) - 1)
            if len(self._buffer) > keep:
                output.append(self._buffer[:-keep] if keep else self._buffer)
                self._buffer = self._buffer[-keep:] if keep else ""
            break

        return "".join(output)

    def finish(self) -> str:
        if self._inside_thinking:
            self._buffer = ""
            return ""
        tail = strip_thinking_content(self._buffer)
        self._buffer = ""
        return tail


class AgentCallError(RuntimeError):
    """大模型调用失败，允许上层 API 明确终止当前任务。"""


def _format_agent_call_error(error: Exception) -> str:
    error_type = type(error).__name__
    error_text = str(error).strip() or "未提供错误详情"
    normalized = f"{error_type} {error_text}".lower()

    if "connection" in normalized or "connect" in normalized:
        return (
            f"无法连接大模型服务 {SERVE_URL}。请检查 WSL 网络、LLM_BASE_URL、代理或证书，"
            "并先运行 `python test_qwen_api.py`。"
        )
    if "authentication" in normalized or "401" in normalized or "api key" in normalized:
        return "大模型服务认证失败。请重新设置 MODAGENT_API_KEY，并确认密钥仍然有效。"
    if "timeout" in normalized:
        return (
            f"连接大模型服务 {SERVE_URL} 超时。请检查服务状态，或适当提高 "
            "LLM_TIMEOUT_SECONDS。"
        )
    if "rate" in normalized or "429" in normalized:
        return "大模型服务触发限流。请稍后重试或检查账户的请求额度。"
    return f"大模型调用失败（{error_type}）：{error_text}"


def _exception_chain(error: Exception) -> str:
    parts = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {str(current).strip()}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def get_default_client():
    """Lazily create the shared client used by difficulty evaluation."""
    global _default_client
    if _default_client is None:
        _default_client = create_llm_client()
    return _default_client


class Agent:
    def __init__(self, client, role_message, role, examplers=None, trace_recorder=None):
        self.role = role
        self.client = client
        self.model_name = MODEL_NAME
        self.trace_recorder = trace_recorder
        self.messages = [
            {"role": "system", "content": role_message},
        ]
        if examplers is not None:
            for exampler in examplers:
                self.messages.append({"role": "user", "content": exampler['question']})
                self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def _record_trace_call(
        self,
        prompt,
        response,
        started_at,
        status,
        error=None,
    ):
        if self.trace_recorder is None:
            return
        try:
            self.trace_recorder.record_agent_call(
                role=self.role,
                prompt=prompt,
                response=response,
                latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                status=status,
                error=error,
            )
        except Exception as trace_error:
            # Trace collection must never make the online medical answer fail.
            print(f"轨迹记录失败: {trace_error}")

    def chat(self, message, callback=None, agent_name=None):
        """
        支持流式输出的聊天方法
        callback: 回调函数，用于处理流式输出 callback(type, content, agent_name)
        agent_name: 可选的智能体名称，用于回调标识
        """
        started_at = time.perf_counter()
        prepared_message = apply_thinking_instruction(message)
        self.messages.append({"role": "user", "content": prepared_message})
        
        # 根据角色选择生成参数
        if self.role == '医学初步评估专家' or self.role == '招募者':
            config = GENERATION_CONFIG_GREEDY.copy()
        else:
            config = GENERATION_CONFIG_BASE.copy()
        
        config["model"] = self.model_name
        config["messages"] = self.messages
        
        try:
            if config.get("stream", False):
                # 流式输出
                response_stream = self.client.chat.completions.create(**config)
                full_response = ""
                callback_buffer = ""
                thinking_filter = ThinkingTagFilter()

                for chunk in response_stream:
                    if not getattr(chunk, "choices", None):
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    
                    if content:
                        visible_content = thinking_filter.feed(content)
                        if not visible_content:
                            continue
                        full_response += visible_content
                        callback_buffer += visible_content

                        # 合并细碎 token 后再发送，避免大量 SSE 事件导致
                        # 浏览器频繁重绘或长连接缓冲区压力过大。
                        if callback and len(callback_buffer) >= 160:
                            callback("incremental", callback_buffer, agent_name or self.role)
                            callback_buffer = ""

                tail = thinking_filter.finish()
                if tail:
                    full_response += tail
                    callback_buffer += tail

                full_response = strip_thinking_content(full_response)
                if callback and callback_buffer:
                    callback("incremental", callback_buffer, agent_name or self.role)
                
                # 流式输出完成后，添加到消息历史
                self.messages.append({"role": "assistant", "content": full_response})
                self._record_trace_call(
                    prompt=prepared_message,
                    response=full_response,
                    started_at=started_at,
                    status="success",
                )
                return full_response
            else:
                # 非流式输出（保持原有逻辑）
                response = self.client.chat.completions.create(**config)
                content = strip_thinking_content(response.choices[0].message.content)
                self.messages.append({"role": "assistant", "content": content})
                self._record_trace_call(
                    prompt=prepared_message,
                    response=content,
                    started_at=started_at,
                    status="success",
                )
                return content
                
        except Exception as e:
            error_message = _format_agent_call_error(e)
            print(f"API调用出错: {error_message}")
            print(f"API原始异常链: {_exception_chain(e)}")
            self._record_trace_call(
                prompt=prepared_message,
                response=None,
                started_at=started_at,
                status="error",
                error=error_message,
            )
            raise AgentCallError(error_message) from e
    
    def temp_responses(self, message, callback=None, agent_name=None):
        """
        支持流式输出的多温度响应方法
        """
        self.messages.append({"role": "user", "content": apply_thinking_instruction(message)})
        responses = {}
        
        for temperature in TEMP_RESPONSE_TEMPERATURES:
            try:
                config = {
                    "model": self.model_name,
                    "messages": self.messages,
                    "temperature": temperature,
                    "max_tokens": TEMP_RESPONSE_MAX_TOKENS,
                    "stream": TEMP_RESPONSE_STREAM,
                }
                
                response_stream = self.client.chat.completions.create(**config)
                full_response = ""
                callback_buffer = ""
                thinking_filter = ThinkingTagFilter()

                for chunk in response_stream:
                    if not getattr(chunk, "choices", None):
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    
                    if content:
                        visible_content = thinking_filter.feed(content)
                        if not visible_content:
                            continue
                        full_response += visible_content
                        callback_buffer += visible_content

                        if callback and len(callback_buffer) >= 160:
                            callback("incremental", callback_buffer, agent_name or self.role)
                            callback_buffer = ""

                tail = thinking_filter.finish()
                if tail:
                    full_response += tail
                    callback_buffer += tail
                full_response = strip_thinking_content(full_response)
                if callback and callback_buffer:
                    callback("incremental", callback_buffer, agent_name or self.role)
                
                responses[temperature] = full_response
                
            except Exception as e:
                print(f"温度 {temperature} 的API调用出错: {e}")
                responses[temperature] = ""
        
        return responses

def remove_json_markers(text):
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:].lstrip()
    if text.endswith('```'):
        text = text[:-3].rstrip()
    
    return text
def determine_difficulty(question, callback=None):
    if callback:
        callback('step', '开始分析问题', '难度评估智能体', ['解析用户输入', '识别关键信息', '确定任务方向'])
    
    difficulty_prompt = """现在，给定如下的医疗查询，您需要确定它的难度/复杂程度：\n{}\n\n\
请从以下选项中选择：
1）简单：查询涉及基础医学知识或常识性问答，通常有标准答案或事实性描述。常见的问题类型涉及疾病定义与分类、常见症状识别、标准检查项目说明以及指南中直接规定的治疗方式或流程。常见的任务类型涉及问答检索、知识补全及医学术语解释。一些简单的日常交流问候也划分到简单
2）中等：查询涉及简单的推理，涉及复杂病因、合并症、诊断路径等。常见的问题类型涉及辅助诊断分析、多种治疗方案选择与对比、多学科因素交叉问题以及风险评估模型的解释与应用。常见的任务类型涉及多来源意见征集、临床路径推荐及交叉知识解释。
3）困难：查询涉及复杂的推理，需要多源知识整合、模型间协作讨论才能完成推理。常见的问题类型涉及个体化诊疗方案制定、病情发展趋势预测与风险评估、医学伦理相关情境判断及不典型病例的综合分析。常见的任务类型涉及病因分析、个案分析推理、诊疗意见推荐。
返回格式如下（请确保是合法 JSON）：
{{ "理由": "你的理由", "决策": "简单"/"中等"/"困难" }}
    """
    
    print(question)
    medical_agent = Agent(
        get_default_client(),
        role_message='你是进行初步评估的医学专家，你的工作是决定医疗查询的难度/复杂程度。', 
        role='医学初步评估专家'
    )
    
    # 使用流式输出
    # response = medical_agent.chat(difficulty_prompt.format(question), callback=callback, agent_name='难度评估智能体')
    response = medical_agent.chat(difficulty_prompt.format(question))
    print(response)
    response=remove_json_markers(response)
    try:
        response_json = json.loads(response)
        ans = ''
        if '简单' in response_json["决策"]:
            ans = '简单'
        elif '中等' in response_json["决策"]:
            ans = '中等'
        elif '困难' in response_json["决策"]:
            ans = '困难'
        
        result_content = f"## 难度评估结果\n\n**问题难度**: {ans}\n\n**评估理由**: {response_json.get('理由', '无详细理由')}"
        if callback:
            callback('output', result_content, '难度评估智能体')
        return ans
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return "简单"  # 默认返回中等难度
