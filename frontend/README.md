# MedScope AI 前端

这是现有多智能体医疗问答服务的零构建前端，支持：

- 病例输入与分析参数设置
- 用户可选择手动指定分析深度，或调用难度评估智能体自动判定
- 专家协作网络和五阶段决策轨迹
- `POST /chat/stream` SSE 事件实时展示
- 专家输出、最终决策与置信度汇总
- 离线演示回放与 JSON 决策记录导出

## 启动

先在项目根目录启动后端：

```bash
python multi_agent.py
```

再启动静态前端：

```bash
python -m http.server 5173 --directory frontend
```

浏览器访问 `http://127.0.0.1:5173`。默认后端接口是
`http://127.0.0.1:50042/chat/stream`，可点击页面右上角的设置按钮修改。

后端未启动时，可以点击“回放演示决策”体验完整交互。

### 长时间推理连接

后端默认每 10 秒发送一次 SSE 注释心跳，避免多智能体长时间推理时连接被
WSL、代理或浏览器作为空闲连接关闭。可通过环境变量调整：

```bash
export SSE_HEARTBEAT_SECONDS=10
```

修改后需要重新启动 `python multi_agent.py`。在 `curl -N` 中看到
`: heartbeat` 属于正常现象，不会显示在页面事件列表中。

## 动态绑定关系

页面使用 `app.js` 中的统一状态对象驱动显示，真实接口与演示回放共用
`handleStreamEvent()`：

- `agent_step`：更新五阶段进度、阶段耗时和事件流；
- `agent_output`：解析专家招募结果，更新专家节点、观点状态与投票；
- `final_result`：更新最终结论、关键依据、置信度、模型和更新时间；
- `complete`：完成所有阶段并恢复页面操作状态；
- `error`：展示接口错误与降级提示。

前端请求中的难度字段：

```json
{
  "enableDifficultyAgent": true,
  "difficulty": null
}
```

`enableDifficultyAgent` 为 `true` 时，后端调用 `determine_difficulty()`；为
`false` 时使用 `difficulty` 中的 `simple`、`medium` 或 `hard`。

运行绑定冒烟测试：

```bash
node frontend/binding.smoke.test.js
```

## 联网检索开关

病例输入区提供“联网检索（逐次开关）”，默认关闭。开启后，请求体会增加：

```json
{
  "enableWebSearch": true
}
```

该开关只影响下一次提交的分析。搜索由后端完成，搜索服务密钥不会下发到浏览器；
关闭时后端不会发起网页搜索。搜索失败会在事件流中显示回退提示，原有本地知识流程
仍会继续执行。
