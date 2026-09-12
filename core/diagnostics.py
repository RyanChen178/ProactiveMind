"""结构化诊断日志 —— JSON 格式输出 + trace_id 关联 + 上下文传播。

对齐 akashic-agent core/common/diagnostic_log 的核心能力：
  - JSONFormatter：每条日志输出一个 JSON 对象，字段可枚举
  - TraceContext：contextvars 实现的 trace_id / span_id 传播
  - bind_context() / clear_context()：临时绑定字段

用法：
  from core.diagnostics import configure_json_logging, get_logger, bind_context

  configure_json_logging(level="INFO")
  log = get_logger(__name__)
  with bind_context(session_id="s1", turn_id="t1"):
      log.info("turn started", extra={"event": "turn_started"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

_TRACE_ID: ContextVar[str | None] = ContextVar("proactivemind_trace_id", default=None)
_SPAN_ID: ContextVar[str | None] = ContextVar("proactivemind_span_id", default=None)
_FIELDS: ContextVar[dict[str, Any]] = ContextVar("proactivemind_fields", default={})

DEFAULT_FIELDS: tuple[str, ...] = (
    "event",
    "flow",
    "phase",
    "session",
    "turn",
    "tick",
    "action",
    "reason",
    "duration_ms",
    "counts",
    "error_type",
    "note",
    "trace_id",
    "span_id",
)


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


def current_span_id() -> str | None:
    return _SPAN_ID.get()


def current_fields() -> dict[str, Any]:
    return dict(_FIELDS.get())


def new_trace_id() -> str:
    return uuid.uuid4().hex


def bind_context(**kwargs: Any) -> _ContextBinder:
    """临时把字段绑定到当前 context，返回可作 with 的对象。"""
    return _ContextBinder(kwargs)


@contextmanager
def trace_span(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """开启一个新的 trace span，自动管理 trace_id / span_id。"""
    outer_trace = _TRACE_ID.get()
    trace_id = outer_trace or new_trace_id()
    span_id = uuid.uuid4().hex[:16]
    fields_with_ids = {
        "trace_id": trace_id,
        "span_id": span_id,
        "span_name": name,
        **fields,
    }
    token_trace = _TRACE_ID.set(trace_id)
    token_span = _SPAN_ID.set(span_id)
    merged_fields = {**_FIELDS.get(), **fields_with_ids}
    token_fields = _FIELDS.set(merged_fields)
    start = time.perf_counter()
    span_info = {"name": name, "trace_id": trace_id, "span_id": span_id}
    try:
        yield span_info
    except Exception as exc:
        merged_fields["error_type"] = type(exc).__name__
        merged_fields["note"] = str(exc)
        raise
    finally:
        merged_fields["duration_ms"] = round(
            (time.perf_counter() - start) * 1000, 2
        )
        _FIELDS.reset(token_fields)
        _SPAN_ID.reset(token_span)
        if outer_trace is None:
            _TRACE_ID.reset(token_trace)
        else:
            _TRACE_ID.set(outer_trace)


@dataclass
class _ContextBinder:
    """用 with 语法绑定 context 字段。

    如果调用方传了 trace_id，会同步设置 _TRACE_ID；
    否则如果当前没有 trace_id，会生成一个新的。
    """

    fields: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> dict[str, Any]:
        outer = _FIELDS.get()
        merged = {**outer, **self.fields}
        self._fields_token = _FIELDS.set(merged)
        # 如果 fields 含 trace_id，同步到 _TRACE_ID；否则保留现有
        if "trace_id" in self.fields:
            self._trace_token = _TRACE_ID.set(self.fields["trace_id"])
        elif _TRACE_ID.get() is None:
            new_id = new_trace_id()
            merged["trace_id"] = new_id
            self._trace_token = _TRACE_ID.set(new_id)
        else:
            self._trace_token = None
        if "span_id" in self.fields:
            self._span_token = _SPAN_ID.set(self.fields["span_id"])
        else:
            self._span_token = None
        return merged

    def __exit__(self, *_args: object) -> None:
        _FIELDS.reset(self._fields_token)
        if self._trace_token is not None:
            _TRACE_ID.reset(self._trace_token)
        if self._span_token is not None:
            _SPAN_ID.reset(self._span_token)


class JsonFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器。

    每条日志输出一个 JSON 对象，字段顺序固定为：
      timestamp / level / logger / message / trace_id / span_id / <structured fields>
    """

    BUILTIN_ATTRS = frozenset(
        {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs", "message",
            "pathname", "process", "processName", "relativeCreated", "stack_info",
            "exc_info", "exc_text", "thread", "threadName", "taskName",
        }
    )

    def __init__(
        self,
        *,
        extra_fields: tuple[str, ...] = DEFAULT_FIELDS,
        include_trace: bool = True,
    ) -> None:
        super().__init__()
        self._extra_fields = extra_fields
        self._include_trace = include_trace

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 合并 record.__dict__ 中的扩展字段
        for key in self._extra_fields:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        # 合并 contextvars 字段
        ctx_fields = _FIELDS.get()
        for key, value in ctx_fields.items():
            if key not in payload:
                payload[key] = value

        # 合并非内置 attrs（即 extra= 传入的）
        for key, value in record.__dict__.items():
            if key in self.BUILTIN_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if self._include_trace:
            trace_id = _TRACE_ID.get()
            span_id = _SPAN_ID.get()
            if trace_id and "trace_id" not in payload:
                payload["trace_id"] = trace_id
            if span_id and "span_id" not in payload:
                payload["span_id"] = span_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


_configured = False


def configure_json_logging(
    level: str = "INFO",
    *,
    stream=None,
    extra_fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> None:
    """把根 logger 配置为输出 JSON 日志。"""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(extra_fields=extra_fields))
    root = logging.getLogger()
    # 清空已有 handler 避免重复
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取一个 logger，方便统一调用约定。"""
    return logging.getLogger(name)


def reset_for_test() -> None:
    """测试辅助：重置全局配置状态。"""
    global _configured
    _configured = False


class LogCapture:
    """测试辅助：在内存中收集日志记录。

    实现：把自定义 handler 直接挂到根 logger，捕获所有 propagate 来的记录。
    """

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None
        self._original_level: int | None = None

    def __enter__(self) -> "LogCapture":
        handler = logging.Handler()
        handler.emit = self._capture  # type: ignore[assignment]
        root = logging.getLogger()
        root.addHandler(handler)
        # 临时降级根 logger level 到 DEBUG 确保 INFO 也被接收
        self._original_level = root.level
        root.setLevel(logging.DEBUG)
        self._handler = handler
        return self

    def __exit__(self, *_args: object) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None
        if self._original_level is not None:
            logging.getLogger().setLevel(self._original_level)

    def _capture(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def texts(self) -> list[str]:
        return [r.getMessage() for r in self.records]


# 触发 contextvars 导入（无副作用）
_ = Mapping