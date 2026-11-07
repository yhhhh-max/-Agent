"""三个内置工具：todo / calculator / search（离线 mock）。"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

from .tools import ToolContext, ToolDefinition, ToolRegistry


TODO_SCHEMA = {
    "type": "object",
    "description": "按 action 选择 add/list/complete/delete 对应的参数分支",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "add"},
                "task": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "required": ["action", "task"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"action": {"type": "string", "const": "list"}},
            "required": ["action"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "complete"},
                "id": {"type": "integer", "minimum": 1},
            },
            "required": ["action", "id"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "delete"},
                "id": {"type": "integer", "minimum": 1},
            },
            "required": ["action", "id"],
            "additionalProperties": False,
        },
    ],
}


def todo_handler(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    action = args["action"]
    if action == "add":
        task = args["task"].strip()
        if not task:
            raise ValueError("任务内容不能只包含空格")
        return {"status": "added", "item": ctx.store.add_todo(ctx.session_id, task)}
    if action == "list":
        items = ctx.store.list_todos(ctx.session_id)
        return {"status": "ok", "count": len(items), "items": items}
    if action == "complete":
        updated = ctx.store.complete_todo(ctx.session_id, args["id"])
        if not updated:
            raise ValueError(f"当前 session 中未找到 Todo id={args['id']}")
        return {"status": "completed", "id": args["id"]}
    if action == "delete":
        deleted = ctx.store.delete_todo(ctx.session_id, args["id"])
        if not deleted:
            raise ValueError(f"当前 session 中未找到 Todo id={args['id']}")
        return {"status": "deleted", "id": args["id"]}
    raise ValueError(f"未知 action：{action}")


BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_ABS_RESULT = 1e12


def safe_calculate(expression: str) -> int | float:
    """只支持四则运算、整除、取模和有限次幂，不使用 eval。"""
    if len(expression) > 200:
        raise ValueError("算式过长")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("算式语法错误") from exc
    if len(list(ast.walk(tree))) > 60:
        raise ValueError("算式结构过于复杂")

    def walk(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("算式中只允许数字")
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
            value = UNARY_OPS[type(node.op)](walk(node.operand))
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 12 or abs(left) > 1e6:
                    raise ValueError("幂运算过大")
                if left < 0 and not float(right).is_integer():
                    raise ValueError("不支持复数计算")
            try:
                value = BIN_OPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ValueError("除数不能为 0") from exc
            except OverflowError as exc:
                raise ValueError("运算溢出") from exc
        else:
            raise ValueError("检测到不被允许的运算符或函数")

        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("结果不是有限实数")
        if abs(value) > MAX_ABS_RESULT:
            raise ValueError("计算结果超出安全上限")
        return value

    result = walk(tree.body)
    if isinstance(result, float):
        result = round(result, 12)
        if result.is_integer():
            return int(result)
    return result


def calculator_handler(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    expression = args["expression"]
    return {"expression": expression, "result": safe_calculate(expression)}


# 固定资料库，仅用于演示。返回 source=mock://，不声称实时联网。
MOCK_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "react",
        "title": "ReAct Agent 设计",
        "content": "ReAct 通过 Reason/Act/Observe 的循环组织工具型 Agent。每次可调用工具并根据观察继续行动。",
        "keywords": "react agent 工具 提示词 规划 思考 行动 观察",
    },
    {
        "id": "python",
        "title": "Python 语言",
        "content": "Python 是通用编程语言，支持标准库的 sqlite3、json、ast 与 unittest。",
        "keywords": "python 编程 代码 标准库",
    },
    {
        "id": "sqlite",
        "title": "SQLite Session",
        "content": "SQLite 是轻量级嵌入式关系数据库，可用 session_id 对事件与 Todo 记录进行隔离。",
        "keywords": "sqlite session 会话 记忆 数据库 持久化",
    },
    {
        "id": "memory",
        "title": "Agent 上下文压缩",
        "content": "把较旧的对话事件提炼为摘要，并在后续 LLM 调用中保留最新原始对话和工具输出。",
        "keywords": "memory 压缩 摘要 上下文 召回",
    },
    {
        "id": "qwen",
        "title": "DashScope / Qwen",
        "content": "DashScope 提供面向千问模型的兼容 Chat Completions 接口，需按 API Key 所在地域配置 base_URL。",
        "keywords": "dashscope qwen 阿里云 百炼 大模型 api",
    },
]


def search_handler(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"].strip().lower()
    if not query:
        raise ValueError("搜索关键词不能只包含空格")
    limit = args.get("limit", 3)
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", query)
    scored = []
    for doc in MOCK_DOCUMENTS:
        corpus = f"{doc['title']} {doc['content']} {doc['keywords']}".lower()
        score = (5 if query in corpus else 0) + sum(2 for token in tokens if token in corpus)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    hits = [
        {
            "title": doc["title"],
            "snippet": doc["content"],
            "source": f"mock://{doc['id']}",
        }
        for _, doc in scored[:limit]
    ]
    return {"query": args["query"], "offline_mock": True, "count": len(hits), "results": hits}


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="todo",
            description="管理当前 session 的待办事项：add 新增，list 查询，complete 完成，delete 删除；跨 session 不共享。",
            parameters=TODO_SCHEMA,
            handler=todo_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="calculator",
            description="安全执行数学表达式，支持 + - * / // % ** 与括号；不支持函数调用或变量。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string", "minLength": 1, "maxLength": 200,
                        "description": "例如 (8 + 4) * 3",
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=calculator_handler,
        )
    )
    registry.register(
        ToolDefinition(
            name="search",
            description="在本地固定样例资料库中模拟搜索，不联网；搜索结果不可作为实时新闻或事实更新。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_handler,
        )
    )
