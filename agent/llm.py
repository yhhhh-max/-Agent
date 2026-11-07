"""DashScope HTTP Chat Completions 接口：不依赖 SDK，不传 tools/function_call。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """模型接口调用或响应结构错误。"""


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        """返回 LLM 的原始文本，由本项目 Parser 自行解析。"""


@dataclass
class DashScopeLLM:
    api_key: str
    base_url: str
    model: str = "qwen-plus"
    timeout_seconds: float = 45.0

    @property
    def endpoint(self) -> str:
        base = self.base_url.strip().rstrip("/")
        if not base.startswith(("https://", "http://")):
            raise LLMError("base_URL 必须是 http(s) 地址")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key or self.api_key == "your_dashscope_api_key_here":
            raise LLMError("请先在 .env 中配置有效的 DASHSCOPE_API_KEY")

        # 只有 model/messages/temperature，刻意不使用工具调用相关 API 字段。
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        req = Request(
            url=self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read(4 * 1024 * 1024)
        except HTTPError as exc:
            try:
                detail = exc.read(800).decode("utf-8", "replace")
            except Exception:
                detail = ""
            raise LLMError(f"DashScope HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"DashScope 网络请求失败：{exc}") from exc

        try:
            data = json.loads(body.decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise LLMError("DashScope 返回结果不包含有效的 choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("DashScope 返回了空文本或非文本消息")
        return content
