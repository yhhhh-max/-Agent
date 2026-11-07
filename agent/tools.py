"""工具注册、Schema 暴露、参数校验及执行调度。"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

from .schema import validate_schema
from .storage import SQLiteStore


class ToolNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    store: SQLiteStore


ToolHandler = Callable[[ToolContext, dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", definition.name):
            raise ValueError("工具名只能是英文字母开头的字母/数字/下划线")
        if definition.name in self._tools:
            raise ValueError(f"工具重复注册：{definition.name}")
        if definition.parameters.get("type") != "object":
            raise ValueError("工具参数 JSON Schema 顶层必须是 object")
        if not definition.description.strip():
            raise ValueError("工具描述不能为空")
        self._tools[definition.name] = definition

    def specs(self) -> list[dict[str, Any]]:
        """只输出开放给 LLM 的名称、描述和 Schema，不泄漏 handler。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": copy.deepcopy(tool.parameters),
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> Any:
        definition = self._tools.get(name)
        if definition is None:
            raise ToolNotFoundError(f"未注册的工具：{name}")
        validate_schema(arguments, definition.parameters)
        return definition.handler(context, arguments)
