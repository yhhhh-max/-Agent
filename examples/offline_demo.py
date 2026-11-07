"""不需 API Key 的端到端演示，演示的是 Runtime/Tool 真执行，LLM 用脚本回放。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent.builtin_tools import register_builtin_tools
from agent.runtime import AgentRuntime
from agent.storage import SQLiteStore
from agent.tools import ToolRegistry


class ScriptedLLM:
    def __init__(self, answers: list[dict]):
        self.answers = iter(answers)

    def complete(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(next(self.answers), ensure_ascii=False)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        store = SQLiteStore(
            Path(temp) / "demo.sqlite3", Path(temp) / "trace.jsonl"
        )
        tools = ToolRegistry()
        register_builtin_tools(tools)
        llm = ScriptedLLM([
            {"type": "tool_call", "tool": "todo",
             "arguments": {"action": "add", "task": "明天做 Agent 项目复盘"},
             "thought": "先把待办保存到当前会话"},
            {"type": "tool_call", "tool": "calculator",
             "arguments": {"expression": "(12+8)*3"}},
            {"type": "final", "content": "已新增待办；(12+8)*3 = 60。"},
            {"type": "tool_call", "tool": "todo", "arguments": {"action": "list"}},
            {"type": "final", "content": "当前会话已有 1 条待办。"},
            {"type": "tool_call", "tool": "todo", "arguments": {"action": "list"}},
            {"type": "final", "content": "新会话目前没有待办。"},
        ])
        agent = AgentRuntime(llm, store, tools)
        a, b = store.create_session(), store.create_session()

        print("Session A:", a)
        print("提问 A:", agent.run(a, "记下明天做 Agent 项目复盘，再算 (12+8)*3"))
        print("追问 A:", agent.run(a, "我现在有多少待办？"))
        print("Session B:", b)
        print("提问 B:", agent.run(b, "我现在有多少待办？"))
        print("Todo A:", store.list_todos(a))
        print("Todo B:", store.list_todos(b))
        print("A 的 Trace 事件数:", len(store.get_events(a)))
        print("B 的 Trace 事件数:", len(store.get_events(b)))


if __name__ == "__main__":
    main()
