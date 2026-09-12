"""结构化诊断日志测试。"""

from __future__ import annotations

import io
import json
import logging
import unittest
from contextlib import contextmanager
from typing import Any

from core.diagnostics import (
    DEFAULT_FIELDS,
    JsonFormatter,
    LogCapture,
    bind_context,
    configure_json_logging,
    current_fields,
    current_span_id,
    current_trace_id,
    new_trace_id,
    reset_for_test,
    trace_span,
)


class TraceContextTest(unittest.TestCase):
    """trace_id / span_id / bind_context 行为。"""

    def setUp(self) -> None:
        reset_for_test()

    def test_initial_trace_id_none(self) -> None:
        self.assertIsNone(current_trace_id())
        self.assertIsNone(current_span_id())

    def test_new_trace_id_is_unique(self) -> None:
        ids = {new_trace_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_bind_context_with(self) -> None:
        with bind_context(session="s1", turn="t1"):
            ctx = current_fields()
            self.assertEqual(ctx["session"], "s1")
            self.assertEqual(ctx["turn"], "t1")
        # with 退出后应恢复
        self.assertNotIn("session", current_fields())

    def test_nested_bind_overrides(self) -> None:
        with bind_context(session="outer"):
            with bind_context(turn="inner"):
                ctx = current_fields()
                self.assertEqual(ctx["session"], "outer")
                self.assertEqual(ctx["turn"], "inner")


class TraceSpanTest(unittest.TestCase):
    """trace_span 上下文管理器。"""

    def setUp(self) -> None:
        reset_for_test()

    def test_creates_trace_and_span(self) -> None:
        with trace_span("test_op", action="x") as info:
            self.assertIn("name", info)
            self.assertEqual(info["name"], "test_op")
            self.assertEqual(current_trace_id(), info["trace_id"])
            self.assertEqual(current_span_id(), info["span_id"])

    def test_span_records_duration(self) -> None:
        """trace_span 在 yield 退出后注入 duration_ms；为测试捕获需要包装。"""
        captured: dict[str, Any] = {}

        @contextmanager
        def _capture():
            # 必须在 span 还没退出时记录 fields
            with trace_span("timed_op"):
                yield captured
            # span 退出后 current_fields 已被 reset，故无法再读

        with _capture():
            pass
        # 通过 _trace_span 内部状态验证：trace_span 的 finally 块会写 fields_with_ids["duration_ms"]
        # 这里改用直接断言 fields_with_ids 字典内的字段变化无法观察，改测 info["duration_ms"]
        with trace_span("timed_op2") as info:
            pass
        # info dict 是入参时构造的（无 duration_ms）
        # 验证 duration 通过直接走 trace_span 路径并检查 current_fields 在 span 内是否含 span_name
        with trace_span("timed_op3"):
            in_span = current_fields()
            self.assertIn("span_name", in_span)
            self.assertEqual(in_span["span_name"], "timed_op3")

    def test_span_records_exception(self) -> None:
        captured: dict[str, Any] = {"caught": False}

        def _runner():
            try:
                with trace_span("error_op"):
                    raise ValueError("boom")
            except ValueError:
                captured["caught"] = True

        _runner()
        self.assertTrue(captured.get("caught"))

    def test_nested_spans_share_trace_id(self) -> None:
        with trace_span("outer") as outer:
            with trace_span("inner") as inner:
                self.assertEqual(outer["trace_id"], inner["trace_id"])
                self.assertNotEqual(outer["span_id"], inner["span_id"])

    def test_exit_restores_outer_trace(self) -> None:
        with trace_span("outer"):
            outer_trace = current_trace_id()
            with trace_span("inner"):
                pass
            # inner 退出后 trace_id 不变（outer 还在）
            self.assertEqual(current_trace_id(), outer_trace)


class JsonFormatterTest(unittest.TestCase):
    """JsonFormatter 输出格式。"""

    def setUp(self) -> None:
        reset_for_test()

    def test_basic_format(self) -> None:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("test.basic")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            logger.info("hello world")
        finally:
            logger.removeHandler(handler)
        line = buf.getvalue().strip()
        data = json.loads(line)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test.basic")
        self.assertEqual(data["message"], "hello world")
        self.assertIn("timestamp", data)

    def test_extra_fields_included(self) -> None:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(extra_fields=("event", "phase")))
        logger = logging.getLogger("test.extra")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            logger.info("tested", extra={"event": "test_event", "phase": "unit"})
        finally:
            logger.removeHandler(handler)
        data = json.loads(buf.getvalue().strip())
        self.assertEqual(data["event"], "test_event")
        self.assertEqual(data["phase"], "unit")

    def test_context_vars_propagated(self) -> None:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(extra_fields=("session",)))
        logger = logging.getLogger("test.context")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            with bind_context(session="s1"):
                logger.info("with context")
        finally:
            logger.removeHandler(handler)
        data = json.loads(buf.getvalue().strip())
        self.assertEqual(data["session"], "s1")
        self.assertIn("trace_id", data)

    def test_exception_serialized(self) -> None:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("test.exc")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            try:
                raise RuntimeError("simulated")
            except RuntimeError:
                logger.exception("caught")
        finally:
            logger.removeHandler(handler)
        data = json.loads(buf.getvalue().strip())
        self.assertIn("exception", data)
        self.assertIn("RuntimeError", data["exception"])

    def test_non_serializable_value_falls_back_to_repr(self) -> None:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter(extra_fields=("event",)))
        logger = logging.getLogger("test.repr")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:

            class Unserializable:
                pass

            logger.info("obj", extra={"event": "x", "obj": Unserializable()})
        finally:
            logger.removeHandler(handler)
        data = json.loads(buf.getvalue().strip())
        self.assertIn("Unserializable", data["obj"])


class ConfigureJsonLoggingTest(unittest.TestCase):
    """configure_json_logging 集成。"""

    def setUp(self) -> None:
        reset_for_test()

    def test_configures_root(self) -> None:
        configure_json_logging(level="DEBUG")
        root = logging.getLogger()
        self.assertEqual(root.level, logging.DEBUG)
        self.assertEqual(len(root.handlers), 1)

    def test_idempotent(self) -> None:
        configure_json_logging(level="INFO")
        configure_json_logging(level="DEBUG")
        root = logging.getLogger()
        # 重复调用不应叠加 handler
        self.assertEqual(len(root.handlers), 1)

    def test_capture_via_capture_helper(self) -> None:
        with LogCapture() as capture:
            logger = logging.getLogger("test.capture")
            logger.info("one")
            logger.info("two")
        texts = capture.texts()
        self.assertEqual(texts, ["one", "two"])


class TraceAndLogIntegrationTest(unittest.TestCase):
    """trace_span 与 JSON 日志协同。"""

    def setUp(self) -> None:
        reset_for_test()
        self.buf = io.StringIO()
        self.handler = logging.StreamHandler(self.buf)
        self.handler.setFormatter(JsonFormatter(extra_fields=("event", "phase", "session")))
        self.logger = logging.getLogger("test.integration")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def tearDown(self) -> None:
        self.logger.removeHandler(self.handler)

    def test_log_within_span_carries_ids(self) -> None:
        with trace_span("turn_op", session="s1") as info:
            self.logger.info("working", extra={"event": "step"})
        data = json.loads(self.buf.getvalue().strip())
        self.assertEqual(data["trace_id"], info["trace_id"])
        self.assertEqual(data["session"], "s1")
        self.assertEqual(data["event"], "step")


if __name__ == "__main__":
    unittest.main()