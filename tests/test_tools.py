import tempfile
import unittest
from pathlib import Path

from agent.builtin_tools import register_builtin_tools, safe_calculate
from agent.schema import SchemaValidationError
from agent.storage import SQLiteStore
from agent.tools import ToolContext, ToolDefinition, ToolRegistry


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = SQLiteStore(root / "db.sqlite3", root / "trace.jsonl")
        self.a = self.store.create_session()
        self.b = self.store.create_session()
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, tool, args, session=None):
        return self.registry.execute(
            tool, args, ToolContext(session or self.a, self.store)
        )

    def test_default_tool_count_and_schema(self):
        self.assertEqual(
            {spec["name"] for spec in self.registry.specs()},
            {"todo", "calculator", "search"},
        )
        self.assertTrue(
            all(spec["parameters"]["type"] == "object" for spec in self.registry.specs())
        )
        self.assertIn("oneOf", next(
            x for x in self.registry.specs() if x["name"] == "todo"
        )["parameters"])

    def test_calculator(self):
        self.assertEqual(self.invoke("calculator", {"expression": "(12 + 8)*3"})["result"], 60)
        self.assertEqual(self.invoke("calculator", {"expression": "7 // 2"})["result"], 3)
        self.assertEqual(safe_calculate("2 ** 5"), 32)
        self.assertAlmostEqual(safe_calculate("0.1+0.2"), 0.3)

    def test_calculator_rejects_code_or_zero_division(self):
        for expr in ('__import__("os").system("echo x")', "1/0", "2**1000", "abs(-5)", "1 << 3"):
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                self.invoke("calculator", {"expression": expr})

    def test_todo_add_list_complete_delete(self):
        added = self.invoke("todo", {"action": "add", "task": "写单元测试"})["item"]
        item_id = added["id"]
        self.assertEqual(self.invoke("todo", {"action": "list"})["count"], 1)
        self.assertFalse(self.invoke("todo", {"action": "list"})["items"][0]["completed"])
        self.assertEqual(self.invoke("todo", {"action": "complete", "id": item_id})["status"], "completed")
        self.assertTrue(self.invoke("todo", {"action": "list"})["items"][0]["completed"])
        self.assertEqual(self.invoke("todo", {"action": "delete", "id": item_id})["status"], "deleted")
        self.assertEqual(self.invoke("todo", {"action": "list"})["count"], 0)

    def test_todo_isolation_at_database_layer(self):
        item_id = self.invoke("todo", {"action": "add", "task": "A 的秘密任务"})["item"]["id"]
        self.assertEqual(self.invoke("todo", {"action": "list"}, session=self.b)["count"], 0)
        with self.assertRaises(ValueError):
            self.invoke("todo", {"action": "delete", "id": item_id}, session=self.b)
        self.assertEqual(self.invoke("todo", {"action": "list"})["count"], 1)

    def test_todo_invalid_schema(self):
        bad = [
            {"action": "add"},
            {"action": "list", "task": "多余字段"},
            {"action": "delete", "id": True},
            {"action": "delete", "id": 0},
            {"action": "complete", "id": "1"},
            {"action": "unknown"},
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(SchemaValidationError):
                self.invoke("todo", args)

    def test_todo_whitespace_and_nonexistent(self):
        with self.assertRaises(ValueError):
            self.invoke("todo", {"action": "add", "task": "  "})
        with self.assertRaises(ValueError):
            self.invoke("todo", {"action": "complete", "id": 999})

    def test_search_is_offline_mock(self):
        result = self.invoke("search", {"query": "Python", "limit": 2})
        self.assertTrue(result["offline_mock"])
        self.assertGreaterEqual(result["count"], 1)
        self.assertTrue(result["results"][0]["source"].startswith("mock://"))
        self.assertEqual(self.invoke("search", {"query": "nonsense_notfound_9382"})["count"], 0)

    def test_duplicate_registration_and_dynamic_extension(self):
        with self.assertRaises(ValueError):
            register_builtin_tools(self.registry)
        self.registry.register(ToolDefinition(
            name="echo",
            description="原样返回",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=lambda ctx, args: {"echo": args["text"]},
        ))
        self.assertEqual(self.invoke("echo", {"text": "你好"}), {"echo": "你好"})


if __name__ == "__main__":
    unittest.main()
