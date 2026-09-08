"""ProactiveMind Python SDK。

通过 JSON-RPC 2.0 over NDJSON 与 ProactiveMind 服务端通信：

  initialize / initialized           握手
  session/list                       列出会话
  session/start                      创建会话
  session/follow                     订阅会话事件流
  session/unfollow                   取消订阅
  turn/start                         发起一轮对话
  turn/read                          读取响应事件流
  turn/interrupt                     中断当前 turn

异步与同步两套客户端共享同一份实现：同步版本通过后台事件循环包装。
"""

from proactivemind_sdk.client import (
    AsyncProactiveMind,
    ConnectionClosedError,
    ProtocolError,
    ProactiveMind,
    RemoteError,
    SessionSubscription,
    SlowConsumerError,
)

__all__ = [
    "AsyncProactiveMind",
    "ConnectionClosedError",
    "ProtocolError",
    "ProactiveMind",
    "RemoteError",
    "SessionSubscription",
    "SlowConsumerError",
]

__version__ = "0.1.0"