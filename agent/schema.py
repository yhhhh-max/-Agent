"""用于工具参数验证的 JSON Schema 常用子集，不依赖第三方库。

支持 type / properties / required / additionalProperties / enum / const /
oneOf / minLength / maxLength / minimum / maximum / items / minItems / maxItems。
本项目所有内置工具的 Schema 都在这个子集内。
"""
from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    pass


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }.get(expected, lambda: False)()


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    if expected and not _type_matches(value, expected):
        raise SchemaValidationError(f"{path}: 参数类型应为 {expected}")

    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise SchemaValidationError(f"{path}: 参数必须是 {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: 不在允许取值 {schema['enum']!r} 内")

    if "oneOf" in schema:
        successful = 0
        reasons = []
        for branch in schema["oneOf"]:
            try:
                validate_schema(value, branch, path)
                successful += 1
            except SchemaValidationError as exc:
                reasons.append(str(exc))
        if successful != 1:
            detail = "; ".join(reasons[:2]) if successful == 0 else "多条分支同时匹配"
            raise SchemaValidationError(f"{path}: 不符合 oneOf 的唯一分支要求。{detail}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaValidationError(f"{path}: 缺少必填字段 {key!r}")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(props)
            if unknown:
                raise SchemaValidationError(f"{path}: 未知字段 {sorted(unknown)!r}")
        for key, item in value.items():
            if key in props:
                validate_schema(item, props[key], f"{path}.{key}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaValidationError(f"{path}: 字符串长度至少 {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(f"{path}: 字符串长度最多 {schema['maxLength']}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: 必须 >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: 必须 <= {schema['maximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaValidationError(f"{path}: 数组至少 {schema['minItems']} 项")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: 数组最多 {schema['maxItems']} 项")
        if "items" in schema:
            for index, element in enumerate(value):
                validate_schema(element, schema["items"], f"{path}[{index}]")
