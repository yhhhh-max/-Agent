"""SQLite Session / Event / Todo 持久化，外加逐事件 JSONL Trace。"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CONTEXT_KINDS = {
    "user", "thought", "tool_call", "tool_result", "tool_error",
    "format_error", "final", "llm_error", "limit_reached",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _short(value: Any, limit: int = 220) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value if len(value) <= limit else value[:limit - 1] + "…"


class SQLiteStore:
    """所有会话相关的读写都必须显式传 session_id，避免窗口串话。"""

    def __init__(self, db_path: Path | str, trace_path: Path | str):
        self.db_path = Path(db_path)
        self.trace_path = Path(trace_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace_lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    turn_counter INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL DEFAULT '',
                    compacted_through INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    turn_number INTEGER NOT NULL,
                    cycle INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_session_id
                    ON events(session_id, id);
                CREATE TABLE IF NOT EXISTS todos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    task TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_todos_session
                    ON todos(session_id, id);
            """)

    def create_session(self, session_id: str | None = None) -> str:
        session_id = session_id or uuid.uuid4().hex
        if not isinstance(session_id, str) or not (1 <= len(session_id) <= 80):
            raise ValueError("session_id 必须是长度 1~80 的非空字符串")
        now = _utc_now()
        with self._connection() as con:
            con.execute(
                "INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connection() as con:
            row = con.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, created_at, updated_at, turn_counter "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def next_turn(self, session_id: str) -> int:
        with self._connection() as con:
            con.execute(
                "UPDATE sessions SET turn_counter=turn_counter+1, updated_at=? WHERE id=?",
                (_utc_now(), session_id),
            )
            row = con.execute(
                "SELECT turn_counter FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"会话不存在：{session_id}")
            return int(row["turn_counter"])

    def add_event(
        self, session_id: str, turn_number: int, cycle: int,
        kind: str, payload: dict[str, Any],
    ) -> int:
        """持久化原始结构化事件；不会因压缩而删除完整审计轨迹。"""
        now = _utc_now()
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._connection() as con:
            cursor = con.execute(
                "INSERT INTO events (session_id, turn_number, cycle, kind, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_number, cycle, kind, encoded, now),
            )
            event_id = int(cursor.lastrowid)

        line = json.dumps(
            {
                "event_id": event_id,
                "timestamp": now,
                "session_id": session_id,
                "turn_number": turn_number,
                "cycle": cycle,
                "kind": kind,
                "payload": payload,
            },
            ensure_ascii=False,
        )
        with self._trace_lock:
            with self.trace_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        return event_id

    def get_events(
        self, session_id: str, after_id: int = 0,
        *, context_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT * FROM events WHERE session_id=? AND id>? ORDER BY id",
                (session_id, after_id),
            ).fetchall()
        events = []
        for row in rows:
            if context_only and row["kind"] not in CONTEXT_KINDS:
                continue
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            events.append(item)
        return events

    def get_context(self, session_id: str) -> tuple[str, list[dict[str, Any]]]:
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"会话不存在：{session_id}")
        events = self.get_events(
            session_id, after_id=session["compacted_through"], context_only=True
        )
        return session["summary"], events

    @staticmethod
    def _event_summary(event: dict[str, Any]) -> str:
        p = event["payload"]
        kind = event["kind"]
        if kind in {"user", "thought", "final"}:
            detail = p.get("content", "")
        elif kind == "tool_call":
            detail = f"{p.get('tool')} args={_short(p.get('arguments', {}), 130)}"
        elif kind in {"tool_result", "tool_error"}:
            detail = f"{p.get('tool')} {_short(p.get('result', p.get('error', '')), 140)}"
        elif kind == "format_error":
            detail = p.get("error", "")
        else:
            detail = p
        return f"[turn {event['turn_number']}][{kind}] {_short(detail)}"

    @classmethod
    def _summarize_completed_turn(
        cls, events: list[dict[str, Any]], max_chars: int,
    ) -> str:
        """整轮摘要占一行，裁剪时保留开头提问和结尾结论。"""
        turn = events[0]["turn_number"]
        fragments = [cls._event_summary(event) for event in events]
        text = f"[completed turn {turn}] " + " | ".join(fragments)
        if len(text) <= max_chars:
            return text
        # 摘要本身有损，但不会只移动半轮原始事件到 compacted_through。
        front = max_chars * 3 // 5
        back = max_chars - front - 5
        return text[:front] + " ... " + text[-back:]

    def maybe_compress(
        self, session_id: str, *, before_turn: int,
        threshold: int = 18, keep_recent: int = 8,
        max_chars: int = 3000,
    ) -> bool:
        """仅压缩当前轮之前的、已结束的**完整用户轮次**。

        压缩检查只应在一个新 user turn 开始时调用。
        - 当前轮号为 before_turn；这一轮的所有事件永远不参与本次压缩。
        - 结束标记必须是 final / limit_reached；不完整的旧轮及其后续事件
          不会被跨越（compacted_through 是单个连续事件位置）。
        - threshold / keep_recent 仍以【上下文事件数】衡量，但 cutoff 只在
          旧轮结束位置落点，绝不会切开 user/tool_call/tool_result/final。
        - 已压缩的原始事件继续保留在 SQLite/JSONL 中用于 Trace 审计。
        """
        if not (threshold > keep_recent >= 1 and max_chars >= 200):
            raise ValueError("需要 threshold > keep_recent >= 1 且 max_chars >= 200")
        if not isinstance(before_turn, int) or isinstance(before_turn, bool):
            raise ValueError("before_turn 必须是当前 session 的整数轮次")

        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"会话不存在：{session_id}")
        if before_turn != session["turn_counter"]:
            raise ValueError("before_turn 必须等于当前 session 的 turn_counter")

        pending = self.get_events(
            session_id, after_id=session["compacted_through"], context_only=True
        )
        if len(pending) <= threshold:
            return False

        # 只能取连续的【已结束旧轮】前缀，不能跳过中间未结束的旧轮。
        old_turns: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(pending):
            turn_number = pending[i]["turn_number"]
            if turn_number >= before_turn:
                break
            j = i + 1
            while j < len(pending) and pending[j]["turn_number"] == turn_number:
                j += 1
            turn_events = pending[i:j]
            # 防止旧版本曾在轮次中间截断后，继续把残余半轮当成完整轮。
            if (turn_events[0]["kind"] != "user"
                    or turn_events[-1]["kind"] not in {"final", "limit_reached"}):
                break
            old_turns.append(turn_events)
            i = j

        # 最多压缩完整旧轮；尽量保证最近 keep_recent 个事件仍是原文。
        selected: list[list[dict[str, Any]]] = []
        selected_event_count = 0
        for turn_events in old_turns:
            candidate = selected_event_count + len(turn_events)
            if len(pending) - candidate < keep_recent:
                break
            selected.append(turn_events)
            selected_event_count = candidate

        if not selected:
            return False

        # 既有摘要逐行保留；新增摘要每轮一行，超限时整行丢弃最老轮。
        # 兼容旧版本可能存储的逐事件摘要行。
        lines = [line for line in session["summary"].splitlines() if line.strip()]
        per_turn_limit = max(100, min(900, max_chars // 2))
        for turn_events in selected:
            lines.append(self._summarize_completed_turn(turn_events, per_turn_limit))
        while lines and len("\n".join(lines)) > max_chars:
            lines.pop(0)

        summary = "\n".join(lines)
        cutoff = selected[-1][-1]["id"]  # 一定为最后一轮 final / limit_reached 的事件 id
        with self._connection() as con:
            con.execute(
                "UPDATE sessions SET summary=?, compacted_through=? WHERE id=?",
                (summary, cutoff, session_id),
            )
        return True

    def add_todo(self, session_id: str, task: str) -> dict[str, Any]:
        with self._connection() as con:
            cursor = con.execute(
                "INSERT INTO todos (session_id, task, created_at) VALUES (?, ?, ?)",
                (session_id, task, _utc_now()),
            )
            item_id = int(cursor.lastrowid)
        return {"id": item_id, "task": task, "completed": False}

    def list_todos(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as con:
            rows = con.execute(
                "SELECT id, task, completed FROM todos WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {"id": int(row["id"]), "task": row["task"], "completed": bool(row["completed"])}
            for row in rows
        ]

    def delete_todo(self, session_id: str, item_id: int) -> bool:
        with self._connection() as con:
            cursor = con.execute(
                "DELETE FROM todos WHERE session_id=? AND id=?",
                (session_id, item_id),
            )
            return cursor.rowcount > 0

    def complete_todo(self, session_id: str, item_id: int) -> bool:
        with self._connection() as con:
            cursor = con.execute(
                "UPDATE todos SET completed=1 WHERE session_id=? AND id=?",
                (session_id, item_id),
            )
            return cursor.rowcount > 0
