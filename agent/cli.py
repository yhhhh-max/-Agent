"""命令行入口：创建/切换 session，继续历史会话、查看 Trace 与 Todo。"""
from __future__ import annotations

import argparse
import json
import sys

from .builtin_tools import register_builtin_tools
from .config import Settings
from .llm import DashScopeLLM
from .runtime import AgentRuntime, RuntimeConfig
from .storage import SQLiteStore
from .tools import ToolRegistry


HELP = """命令：
  /new                 创建并切换到一个新 session
  /sessions            查看已有 session
  /switch SESSION_ID   切换到已有 session
  /todos               查看当前 session 的 Todo（不耗模型 Token）
  /trace [N]           查看当前 session 最近 N 条事件（默认 12）
  /id                  显示当前 session_id
  /help                显示帮助
  /exit                退出
其余输入均发送给 Agent；每次提问最多运行 max_cycles 轮。
"""


def create_runtime(settings: Settings, max_cycles: int) -> tuple[AgentRuntime, SQLiteStore]:
    store = SQLiteStore(
        settings.data_dir / "sessions.sqlite3",
        settings.data_dir / "agent_trace.jsonl",
    )
    registry = ToolRegistry()
    register_builtin_tools(registry)
    llm = DashScopeLLM(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        timeout_seconds=settings.timeout_seconds,
    )
    runtime = AgentRuntime(
        llm, store, registry, RuntimeConfig(max_cycles=max_cycles)
    )
    return runtime, store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="自实现 Mini ReAct Agent")
    parser.add_argument("--session", help="恢复已有 session_id")
    parser.add_argument("--once", help="执行一次输入并退出（方便演示或脚本调用）")
    parser.add_argument("--max-cycles", type=int, default=8)
    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if not settings.api_key or settings.api_key == "your_dashscope_api_key_here":
        print("错误：请先在项目根目录 .env 中填写有效 DASHSCOPE_API_KEY。", file=sys.stderr)
        return 2
    try:
        runtime, store = create_runtime(settings, args.max_cycles)
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.session:
        if not store.get_session(args.session):
            print(f"错误：找不到 session_id={args.session}", file=sys.stderr)
            return 2
        session_id = args.session
    else:
        session_id = store.create_session()

    print(f"✅ Agent 已启动，session_id={session_id}")
    if args.once is not None:
        print(runtime.run(session_id, args.once))
        return 0

    print(HELP)
    while True:
        try:
            line = input(f"[{session_id[:8]}] 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            print("已退出。")
            return 0
        if line == "/help":
            print(HELP)
            continue
        if line == "/new":
            session_id = store.create_session()
            print(f"已创建并切换：{session_id}")
            continue
        if line == "/id":
            print(session_id)
            continue
        if line == "/sessions":
            for session in store.list_sessions():
                marker = "*" if session["id"] == session_id else " "
                print(f"{marker} {session['id']}  turns={session['turn_counter']}")
            continue
        if line.startswith("/switch "):
            target = line.removeprefix("/switch ").strip()
            if store.get_session(target):
                session_id = target
                print(f"已切换到：{session_id}")
            else:
                print("session_id 不存在。")
            continue
        if line == "/todos":
            print(json.dumps(store.list_todos(session_id), ensure_ascii=False, indent=2))
            continue
        if line.startswith("/trace"):
            parts = line.split()
            try:
                n = max(1, min(100, int(parts[1]))) if len(parts) > 1 else 12
            except ValueError:
                print("用法：/trace [数字]")
                continue
            for event in store.get_events(session_id)[-n:]:
                print(json.dumps({
                    "turn": event["turn_number"],
                    "cycle": event["cycle"],
                    "kind": event["kind"],
                    "payload": event["payload"],
                }, ensure_ascii=False))
            continue
        if line.startswith("/"):
            print("未知命令，输入 /help 查看帮助。")
            continue
        try:
            print("Agent >", runtime.run(session_id, line))
        except Exception as exc:
            print(f"系统错误：{type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
