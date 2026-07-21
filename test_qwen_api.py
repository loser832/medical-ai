"""Connectivity and authentication check for the online Qwen endpoint."""

from config import GENERATION_CONFIG_GREEDY, MODEL_NAME, SERVE_URL
from llm_client import create_llm_client
from utils import apply_thinking_instruction, strip_thinking_content


def run_probe(client, name, messages, config=None) -> str:
    request_config = (config or {}).copy()
    request_config.update({
        "model": MODEL_NAME,
        "messages": messages,
    })
    completion = client.chat.completions.create(**request_config)
    content = strip_thinking_content(completion.choices[0].message.content)
    if not content:
        raise RuntimeError(f"{name}返回了空内容")
    print(f"{name}成功。")
    return content


def main() -> None:
    print(f"检查大模型服务：{SERVE_URL}")
    print(f"检查模型：{MODEL_NAME}")
    client = None
    try:
        client = create_llm_client()
        minimal_content = run_probe(
            client,
            "最小 API 健康检查",
            [
                {
                    "role": "user",
                    "content": apply_thinking_instruction("请用一句话说明你当前使用的模型。"),
                },
            ],
        )
        recruiter_content = run_probe(
            client,
            "专家招募参数检查",
            [
                {
                    "role": "system",
                    "content": "你是医疗多智能体系统的专家招募者。",
                },
                {
                    "role": "user",
                    "content": apply_thinking_instruction(
                        "请为急性脑卒中溶栓评估招募3名专家，每行输出专家名称、专长和层级结构。"
                    ),
                },
            ],
            GENERATION_CONFIG_GREEDY,
        )
    except Exception as error:
        error_type = type(error).__name__
        error_text = str(error).strip() or "未提供错误详情"
        normalized = f"{error_type} {error_text}".lower()
        print(f"API 健康检查失败（{error_type}）：{error_text}")
        if "connection" in normalized or "connect" in normalized:
            print("建议检查：WSL 网络、DNS、HTTPS 证书、代理设置以及 LLM_BASE_URL。")
        elif "401" in normalized or "authentication" in normalized or "api key" in normalized:
            print("建议检查：MODAGENT_API_KEY 是否正确、是否过期或已被撤销。")
        elif "timeout" in normalized:
            print("建议检查：服务是否可达，或提高 LLM_TIMEOUT_SECONDS。")
        raise SystemExit(1) from error
    finally:
        if client is not None:
            client.close()

    print("全部 API 健康检查成功。")
    print(minimal_content)
    print(recruiter_content)


if __name__ == "__main__":
    main()
