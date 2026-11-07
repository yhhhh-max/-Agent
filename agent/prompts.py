"""提示词模板只写协议，工具定义全部从注册表动态注入。"""
from __future__ import annotations

import json
from pathlib import Path

from .tools import ToolRegistry


TEMPLATE_FILE = Path(__file__).resolve().parent.parent / "SYSTEM_PROMPT.md"
PLACEHOLDER = "{{TOOLS_JSON}}"


def build_system_prompt(registry: ToolRegistry) -> str:
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"系统提示词缺少动态插入标记：{PLACEHOLDER}")
    specs_json = json.dumps(
        registry.specs(), ensure_ascii=False, indent=2, sort_keys=True
    )
    return template.replace(PLACEHOLDER, specs_json)
