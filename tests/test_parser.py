import unittest

from agent.parser import ParseError, parse_llm_output
from agent.schema import SchemaValidationError, validate_schema


class ParserTests(unittest.TestCase):
    def test_thought(self):
        result = parse_llm_output('{"type":"thought","content":"先整理上下文"}')
        self.assertEqual(result.type, "thought")
        self.assertEqual(result.content, "先整理上下文")

    def test_tool_call(self):
        result = parse_llm_output(
            '{"type":"tool_call","tool":"calculator","arguments":{"expression":"2*7"},"thought":"计算"}'
        )
        self.assertEqual(result.type, "tool_call")
        self.assertEqual(result.tool, "calculator")
        self.assertEqual(result.arguments, {"expression": "2*7"})
        self.assertEqual(result.thought, "计算")

    def test_final(self):
        self.assertEqual(
            parse_llm_output('{"type":"final","content":"你好！"}').content, "你好！"
        )

    def test_markdown_json_fence(self):
        result = parse_llm_output('```json\n{"type":"final","content":"OK"}\n```')
        self.assertEqual(result.content, "OK")

    def test_invalid_json(self):
        for raw in ("hello", "{bad json}", "", "[]", '{"type":"tool_call"}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ParseError):
                    parse_llm_output(raw)

    def test_invalid_type_extra_fields_and_arguments(self):
        bad_outputs = [
            '{"type":"unknown","content":"x"}',
            '{"type":"final","content":"ok","extra":"x"}',
            '{"type":"tool_call","tool":"todo","arguments":[]}',
            '{"type":"tool_call","tool":"todo","arguments":null}',
            '{"type":"final","content":""}',
            '{"type":"thought","content":"ok","thought":"extra"}',
            'answer: {"type":"final","content":"ok"}',
        ]
        for raw in bad_outputs:
            with self.subTest(raw=raw), self.assertRaises(ParseError):
                parse_llm_output(raw)

    def test_duplicate_json_keys_and_nan(self):
        for raw in (
            '{"type":"final","type":"final","content":"ok"}',
            '{"type":"tool_call","tool":"calculator","arguments":{"expression":NaN}}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ParseError):
                parse_llm_output(raw)


class SchemaTests(unittest.TestCase):
    def test_required_unknown_type_and_bounds(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "minimum": 1},
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["id"],
            "additionalProperties": False,
        }
        validate_schema({"id": 1, "mode": "a"}, schema)
        for invalid in (
            {}, {"id": 0}, {"id": True}, {"id": "1"},
            {"id": 1, "foo": 2}, {"id": 1, "mode": "c"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(SchemaValidationError):
                validate_schema(invalid, schema)

    def test_array_items(self):
        schema = {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {"type": "integer"},
        }
        validate_schema([1, 2], schema)
        for invalid in ([], [1, 2, 3], [1, "2"]):
            with self.subTest(invalid=invalid), self.assertRaises(SchemaValidationError):
                validate_schema(invalid, schema)


if __name__ == "__main__":
    unittest.main()
