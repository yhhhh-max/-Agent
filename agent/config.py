"""环境配置：不依赖 python-dotenv，并避免把密钥硬编码进代码。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """轻量加载 KEY=VALUE；现有系统环境变量优先。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str = "qwen-plus"
    data_dir: Path = PROJECT_ROOT / "data"
    timeout_seconds: int = 45

    @classmethod
    def from_env(cls, root: Path = PROJECT_ROOT) -> "Settings":
        load_dotenv(root / ".env")
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=(
                os.getenv("base_URL")
                or os.getenv("BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ).strip(),
            model=os.getenv("DASHSCOPE_MODEL", "qwen-plus").strip(),
            data_dir=root / "data",
        )
