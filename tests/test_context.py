"""上下文历史视图测试。"""

from __future__ import annotations

import unittest

from agent.context import build_history_view, estimate_message_tokens


class ContextTest(unittest.TestCase):
    def test_keeps_complete_tool_call_group(self) -> None:
        tool_call = {
            "id": "call-time",
            "type": "function",
            "function": {"name": "get_time", "arguments": "{}"},
        }
        messages = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call-time", "content": "10:00"},
            {"role": "assistant", "content": "现在十点"},
        ]
        budget = sum(estimate_message_tokens(message) for message in messages[1:])

        view = build_history_view(messages, budget)

        self.assertEqual(view, messages[1:])

    def test_discards_whole_old_groups_when_budget_is_small(self) -> None:
        messages = [
            {"role": "user", "content": "很早之前的问题"},
            {"role": "assistant", "content": "很早之前的回答"},
            {"role": "user", "content": "最新问题"},
        ]
        budget = estimate_message_tokens(messages[-1])

        self.assertEqual(build_history_view(messages, budget), [messages[-1]])

    def test_keeps_latest_group_when_it_exceeds_budget(self) -> None:
        message = {"role": "user", "content": "很长" * 100}

        self.assertEqual(build_history_view([message], 1), [message])

    def test_rejects_non_positive_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            build_history_view([], 0)


if __name__ == "__main__":
    unittest.main()
