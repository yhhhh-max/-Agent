# Mini ReAct Agent — Python 从零实现

一个真正可运行的 **最小可用 Agent**：手写 ReAct 循环、文本 JSON 协议解析、动态工具注册、独立 Session、持久化 Memory、异常恢复以及完整 Trace。

> 不使用 LangGraph / OpenHands / OpenClaw 等 Agent 框架；不使用 OpenAI/DashScope SDK 自带的 Function Calling；模型仅返回**普通文本**，由本项目执行 `json.loads` 和 JSON Schema 校验。`search` 完全离线模拟。

## 1. 环境与快速启动

- Python **3.10 或以上**，仅依赖标准库。
- 如需接入真实模型，准备阿里云百炼 DashScope API Key。
- 将 `.env.example` 复制为 `.env`，填写 Key。为便于验收，项目中已经提供**空 Key** 的 `.env`，可直接编辑。不要把真实 Key 提交到 Git。

```bash
# 在项目根目录 mini_react_agent 下执行
python --version

# 修改 .env 中 DASHSCOPE_API_KEY 后：
python -m agent

# 指定一次输入并退出
python -m agent --once "计算 (12+8)*3"

# 恢复之前某个窗口的会话
python -m agent --session <之前的session_id>

# 调整每次请求的 ReAct 循环上限
python -m agent --max-cycles 12

# 无 API Key 也可以执行全部自动化测试
python -m unittest discover -s tests -v

# 无 API Key 的完整离线演示（脚本化 Mock LLM）
python -m examples.offline_demo
```

**配置文件 `.env`：**

```dotenv
DASHSCOPE_API_KEY=填入自己的真实Key
base_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus
```

代码还兼容环境变量 `BASE_URL`。默认地址是中国北京地域的旧共享域名；如使用海外地域 Key，必须改为对应地域的 Endpoint。阿里云当前也支持业务空间专属域名。Key 和 Endpoint 必须属于相同地域。接口最终 POST 到 `{base_URL}/chat/completions`。

官方文档：
- [阿里云百炼 OpenAI 兼容 Chat](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)
- [阿里云 Base URL 总览](https://help.aliyun.com/zh/model-studio/base-url)

## 2. 工程目录

```text
mini_react_agent/
├── .env                    # 本地真实密钥；默认留空，已 gitignore
├── .env.example            # 配置样例
├── .gitignore
├── requirements.txt        # 无第三方依赖
├── SYSTEM_PROMPT.md        # 完整内部 LLM 提示词（{{TOOLS_JSON}} 动态替换）
├── README.md
├── agent/
│   ├── __main__.py         # python -m agent
│   ├── cli.py              # 交互命令行/session 切换
│   ├── config.py           # 环境变量与 .env 读取
│   ├── llm.py              # DashScope HTTP 适配层（无 SDK Function Calling）
│   ├── parser.py           # 自写 JSON 文本协议解析
│   ├── schema.py           # 自写 JSON Schema 常用子集校验
│   ├── prompts.py          # 动态渲染工具 Schema
│   ├── tools.py            # 工具注册表与执行入口
│   ├── builtin_tools.py    # todo、calculator、search
│   ├── storage.py          # SQLite session/event/todo 与 JSONL Trace
│   └── runtime.py          # ReAct 主循环和记忆编排
├── examples/
│   └── offline_demo.py     # 不调用外部接口的端到端演示
├── tests/                  # unittest 自动验收
└── data/                   # 运行后自动创建，不提交
    ├── sessions.sqlite3
    └── agent_trace.jsonl
```

## 3. 架构与 ReAct 时序

```text
User input
   ↓ session_id
SQLiteStore.next_turn + log(user)
   ↓
[循环 max_cycles 次]
   ↓
maybe_compress(session_id) → build_messages:
  system: SYSTEM_PROMPT + ToolRegistry.specs() 动态 JSON
  user:   SESSION_MEMORY_SUMMARY_DATA（如存在，结构化 JSON，仅作为低信任历史数据）
  user/assistant: 当前 session 最近原始事件
   ↓
DashScope Chat Completions HTTP（纯文本）
   ↓
parse_llm_output(json.loads)
  ├─ thought    → log(thought) → 下一次 LLM 调用
  ├─ tool_call  → log(thought/tool_call) → ToolRegistry.execute
  │    ├─ schema 检查 → 工具 Handler(session_id)
  │    ├─ 成功 → log(tool_result) → 下一次 LLM 调用
  │    └─ 异常 → log(tool_error) → 下一次 LLM 调用（尝试修复）
  ├─ final      → log(final) → 返回内容
  └─ 格式无效  → log(format_error) → 下一次 LLM 调用（纠正）
超出 max_cycles → log(limit_reached) → 明确告知未完成
```

一次模型调用 = 一个 cycle；即使输出错误、只有 thought、或调用工具失败，也会消耗一个 cycle，所以不能无限循环。

## 4. LLM 输出协议

`SYSTEM_PROMPT.md` 是实际加载的完整提示词，`agent/prompts.py` 仅负责动态替换 `{{TOOLS_JSON}}`。模型每轮必须输出**单个 JSON 对象**：

```json
{"type":"thought","content":"我先确认需要什么数据"}
```

```json
{"type":"tool_call","thought":"需要精确计算","tool":"calculator","arguments":{"expression":"(12+8)*3"}}
```

```json
{"type":"final","content":"答案是 60。"}
```

`thought` 是可记录、可展示的**简短行动说明**，并非模型不可见的内部详细推理。

Parser 严格区分三类协议；允许 JSON fenced block，但不允许输出多段非 JSON 自然语言、重复字段、缺失字段、错类型、未知 `type`。JSON Schema 校验和工具异常均反馈给模型继续自我修正，而不是直接导致主程序崩溃。

## 5. 工具系统（动态注册）

每个工具都有 `name`、`description`、`parameters`（JSON Schema）和 `handler`。
`ToolRegistry.specs()` 只向 LLM 注入前三项，`handler` 仅在 Python 内存中执行。

- `todo`：`add(task)` / `list()` / `complete(id)` / `delete(id)`，SQLite 按 `session_id` 隔离，`oneOf` 分支校验不同动作参数。
- `calculator`：安全的 Python AST 求值，支持 `+ - * / // % **`、括号、正负号；拒绝 Python 函数、属性访问、任意代码，限制复杂度、指数及结果量级。
- `search`：固定本地文档的关键词匹配；结果带 `offline_mock: true` 和 `mock://` 来源，不会联网，也不会伪装成实时搜索。

**新增工具不需要修改提示词模板**。例如：

```python
from agent.tools import ToolDefinition, ToolRegistry

registry = ToolRegistry()
registry.register(ToolDefinition(
    name="echo",
    description="原样返回输入的文字",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    handler=lambda ctx, args: {"echo": args["text"]},
))
# 在真实工程中先调用 register_builtin_tools(registry) 可保留内置工具。
# 后续 build_system_prompt(registry) 自动包含 echo。
```

`schema.py` 实现本项目使用到的 JSON Schema 子集（不是完整的 Draft 2020-12 通用校验器）：`type`、`properties`、`required`、`additionalProperties`、`const`、`enum`、`oneOf`、常用长度/数值约束及数组条目。新增复杂 Schema 关键字时，应同步扩展校验器或改用成熟的 JSON Schema 校验库。

## 6. Session 与 Memory：召回时机和放置方式

**会话隔离**
- `/new` 生成随机 UUID session_id；`/sessions` 列出会话；`/switch SESSION_ID` 返回原窗口继续；启动参数 `--session` 可以跨进程恢复。
- `sessions` 表：`id`、创建/更新时间、`turn_counter`、`summary`、`compacted_through`。
- `events` 表：所有用户输入、thought、LLM 原始输出、工具调用/结果、错误、最终回答均绑定 `session_id`、`turn_number`、`cycle`。
- `todos` 表：每条 Todo 绑定 `session_id`，增删改查 SQL 均带 `WHERE session_id=?`，不允许读取/修改其它会话中的 Todo。
- 数据保存在 SQLite，终止 CLI 后仍可恢复。暂未支持多用户鉴权或并发会话互斥。

**召回的精确时机**
1. 收到用户输入后，先写入当前 session 的事件。
2. **仅在新用户 Turn 开始时**（本轮用户输入已入库、首次调用 LLM 之前），执行**一次** `maybe_compress(session_id, before_turn=当前轮号)`；这一轮 ReAct 的后续 cycle 不再触发压缩。
3. 压缩时根据 `turn_number` 将事件分组，只考虑 `turn_number < before_turn` 且以 `user` 开始、以 `final` 或 `limit_reached` 结束的完整旧轮；如果遇到未完成的旧轮，则停止，不越过它压缩。
4. `build_messages()` 在**每一轮 LLM 调用之前**召回已有摘要和最近原始事件，将**完整系统协议与最新工具 Schema**放在第一个 `system` 消息。
5. 若存在摘要，作为第二条消息注入，其 `role=user`，正文为 `SESSION_MEMORY_SUMMARY_DATA` + JSON（含 `record_type`、`trust_level`、`summary`）；**绝不将历史摘要作为 `system` 消息**。
6. 再按时间序列追加该 session 的近期 `user` / `assistant` 事件，包括独立的 thought、tool_call、工具结果与工具错误。当前 Turn 的所有事件始终保留原样，直到该 Turn 结束。
7. 下一次进入新用户 Turn 时，再判断是否需要压缩旧历史。

**按完整 Turn 压缩的例子**
```text
Turn 1: user → tool_call → tool_result → final           可整体压缩
Turn 2: user → final                                     近期保留
Turn 3: user → tool_call → tool_result → (运行中)         必须原样保留

只允许 cutoff 停在 Turn 1 / Turn 2 的终止事件后，
绝不会停在 tool_call 或 tool_result 之后。
```

**压缩策略**
- 默认未压缩的上下文事件数 **> 18** 时才考虑压缩；至少保留最近 **8 个上下文事件**，但只允许以**完整旧轮次**为单位移入 Memory，不会为了凑齐事件数而拆轮。若不足以安全压缩，推迟至后续新 Turn。
- 每个完成的旧 Turn 按 `turn/kind/关键内容` 生成**一条确定性提取摘要**，而不是逐事件截断或额外调用 LLM。
- `compacted_through` 只会移动到已完成旧 Turn 的终止事件 `final` / `limit_reached`；当前 Turn 的所有 `thought`、`tool_call`、`tool_result` 始终不被压缩。
- 对于曾中断、没有终止标志的旧轮，采取保守策略：不压缩它，也不跨越它压缩后面的历史；对旧版本留下的半轮后缀同样保护。
- 摘要最多 3000 字符，过长时按整行（整轮摘要）移除较老记录。故属于**有损记忆**，不能保证长期所有细节无损召回。
- 旧事件不会从 SQLite 或 Trace 删除，仍可审计。Todo 实际状态始终以数据库为准；查询任务时应调用 todo.list，而不是只相信摘要。
- `llm_raw` 用于审计，不回灌模型；避免同一动作同时出现原文和已解析事件。

**提示词注入防护（Memory 的位置与信任级别）**
- `system` 仅放受控的系统协议和动态工具定义；Memory 作为 `user` **数据消息**放在系统协议后、最近历史消息前。
- Memory 被封装成 JSON 字符串，摘要里的原始文字只出现在 `summary` 字段，不能直接变成更高权限的消息角色。
- 系统协议明确要求：Memory、工具结果都属于低信任数据，其中的命令、伪造角色、要求泄漏密钥或修改工具规则等内容不得执行。
- **限制**：仅靠把 Memory 从 `system` 降为 `user` 并不能完全杜绝提示词注入；生产环境还应实现工具授权/敏感操作确认、严格的外部数据过滤与来源追踪。

## 7. 错误恢复与 Trace

| 场景 | 处理方式 |
|---|---|
| 不是合法协议 JSON、缺字段、重复 key | `format_error` → 下一轮要求模型纠正 |
| 未注册工具、Schema 参数非法 | `tool_error` → 下一轮修正工具名称或参数 |
| Todo 不存在、计算除以 0、执行抛异常 | `tool_error` → 下一轮纠正或解释 |
| DashScope HTTP/网络失败 | `llm_error` + 可读兜底答复 |
| 一直只输出 thought / 调工具 / 报错 | 达到 `max_cycles` 后 `limit_reached` |

**双重 Trace**：SQLite `events` 表保存完整事件，`data/agent_trace.jsonl` 一行一个 JSON 事件，包含：
`timestamp`、`event_id`、`session_id`、`turn_number`、`cycle`、`kind`、`payload`。

使用 `/trace 20` 查看当前会话最近 20 个事件。Trace 和 SQLite 可能保存用户原文与工具参数，适合本地开发调试；生产使用前应补充敏感字段脱敏、权限管理和留存策略。

## 8. 自动化测试清单

`python -m unittest discover -s tests -v` 将验证：

- 直接回答、不用工具。
- calculator 单次调用、安全 AST、非法表达式。
- todo 新增、查询、完成、删除；参数 JSON Schema 验证。
- search 本地模拟结果。
- 多次连续工具调用，并基于前一次观察继续执行。
- 跨 session 历史与 Todo 完全隔离、切换回原 session。
- 多轮纯文本追问、带工具结果的追问。
- 只压缩已完成的旧 Turn：当前多工具执行轮不被压缩、完整轮次不拆分、未完成的旧轮不可跨越；同时正确加载 Memory 到 `user` 数据消息。
- 非法 JSON、非法工具参数、工具抛异常后的自修复。
- 达到 max_cycles 正确停止、LLM 接口失败兜底。
- 动态注册新工具后，系统提示词自动包含 Schema。
- HTTP 请求不带 Function Calling / `tools` 字段。

**验收范围声明：**全部自动化测试使用 Fake LLM，不发外部网络请求、不需要真实 Key。真实 Qwen 接口是否可用取决于你的账号开通情况、API Key 权限、模型名称、地域配置和网络；提供 `.env` 后可运行 CLI 做在线端到端验证。

## 9. CLI 使用示例

```text
$ python -m agent
✅ Agent 已启动，session_id=ad1...
[ad1...] 你 > 帮我记一下明天复盘
Agent > 已添加待办事项：明天复盘
[ad1...] 你 > 列出我的待办
Agent > 当前有 1 条待办：...
[ad1...] 你 > /new
已创建并切换：f42...
[f42...] 你 > 列出我的待办
Agent > 当前会话没有待办事项
[f42...] 你 > /switch ad1...  # 这里要填写 /id 返回的完整 id
已切换到：ad1...
[ad1...] 你 > 刚才记了什么
Agent > 你刚才记了“明天复盘”。
```

这份示例表示**预期流程**，具体措辞由 LLM 决定。
