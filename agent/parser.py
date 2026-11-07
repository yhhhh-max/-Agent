"""显式解析 thought / tool_call / final；完全不依赖 SDK Function Calling。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


class ParseError(ValueError):
    """LLM 没有遵守约定的文本 JSON 协议。"""


ActionType = Literal["thought", "tool_call", "final"]


@dataclass(frozen=True)
class ParsedAction:
    type: ActionType
    content: str = ""
    tool: str = ""
    arguments: dict[str, Any] | None = None
    thought: str = ""


def _reject_constant(token: str) -> None:
    raise ValueError(f"非法 JSON 数字：{token}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复的 JSON key：{key}")
        result[key] = value
    return result


def parse_llm_output(raw: str) -> ParsedAction:
    if not isinstance(raw, str) or not raw.strip():
        raise ParseError("模型输出为空")
    text = raw.strip()
    # 容忍完整的 ```json ... ``` 包装，但拒绝其它自然语言前缀/后缀。
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys
        )
    except (ValueError, TypeError) as exc:
        raise ParseError(f"模型输出不是有效单个 JSON 对象：{exc}") from exc

    if not isinstance(obj, dict):
        raise ParseError("JSON 顶层必须是对象")
    kind = obj.get("type")
    if kind not in ("thought", "tool_call", "final"):
        raise ParseError("type 只能为 thought / tool_call / final")

    allowed = {
        "thought": {"type", "content"},
        "tool_call": {"type", "tool", "arguments", "thought"},
        "final": {"type", "content", "thought"},
    }[kind]
    extra = set(obj) - allowed
    if extra:
        raise ParseError(f"存在未允许的字段：{sorted(extra)}")
    thought = obj.get("thought", "")
    if not isinstance(thought, str) or len(thought) > 1000:
        raise ParseError("thought 必须是长度不超过 1000 的字符串")

    if kind in ("thought", "final"):
        content = obj.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ParseError(f"{kind}.content 必须是非空字符串")
        if len(content) > 20000:
            raise ParseError("content 长度不得超过 20000")
        return ParsedAction(type=kind, content=content.strip(), thought=thought.strip())

    tool = obj.get("tool")
    arguments = obj.get("arguments")
    if not isinstance(tool, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tool):
        raise ParseError("tool 必须是合法的已注册工具名称")
    if not isinstance(arguments, dict):
        raise ParseError("tool_call.arguments 必须是 JSON 对象")
    return ParsedAction(type="tool_call", tool=tool, arguments=arguments, thought=thought.strip())
