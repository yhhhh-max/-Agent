import json
import unittest
from unittest.mock import patch

from agent.llm import DashScopeLLM, LLMError


class LLMTests(unittest.TestCase):
    def test_http_uses_only_text_protocol_not_function_calling(self):
        llm = DashScopeLLM(
            api_key="unit-test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
        )
        response = {
            "choices": [
                {"message": {"role": "assistant",
                             "content": '{"type":"final","content":"OK"}'}}
            ]
        }
        with patch("agent.llm.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                json.dumps(response).encode("utf-8")
            )
            answer = llm.complete([{"role": "user", "content": "hi"}])
        self.assertIn('"type":"final"', answer)
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn("messages", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("functions", payload)
        self.assertNotIn("function_call", payload)

    def test_missing_key_fails_without_network(self):
        llm = DashScopeLLM("", "https://example.com/v1")
        with self.assertRaises(LLMError):
            llm.complete([{"role": "user", "content": "hi"}])

    def test_endpoint_accepts_full_path(self):
        url = "https://example.com/v1/chat/completions"
        self.assertEqual(DashScopeLLM("fake", url).endpoint, url)


if __name__ == "__main__":
    unittest.main()
