"""流式 Provider 的纯解析逻辑测试。"""

from __future__ import annotations

import unittest

from mind.provider import LLMProvider


class StreamToolCallTest(unittest.TestCase):
    def test_merges_split_tool_call_deltas(self) -> None:
        raw_calls: dict[int, dict[str, str]] = {}
        LLMProvider._merge_tool_call_deltas(
            raw_calls,
            [
                {
                    "index": 0,
                    "id": "call_",
                    "function": {"name": "get_", "arguments": '{"zo'},
                },
                {
                    "index": 0,
                    "id": "time",
                    "function": {"name": "time", "arguments": 'ne":"UTC"}'},
                },
            ],
        )

        calls = LLMProvider._parse_stream_tool_calls(raw_calls)

        self.assertEqual(calls[0].id, "call_time")
        self.assertEqual(calls[0].name, "get_time")
        self.assertEqual(calls[0].arguments, {"zone": "UTC"})


if __name__ == "__main__":
    unittest.main()
