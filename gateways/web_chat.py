"""Web Chat 渠道 —— FastAPI + WebSocket 对话界面。

启动后访问 http://127.0.0.1:6322 在浏览器中对话。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from mind.loop import MindLoop

log = logging.getLogger(__name__)


class SocketHub:
    """管理 WebSocket 连接，支持向所有客户端广播。"""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, content: str) -> None:
        """向所有连接的客户端推送主动消息。"""
        message = json.dumps({"type": "proactive", "content": content})
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self._connections)

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProactiveMind</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; }
  #chat { max-width: 720px; margin: 0 auto; padding: 16px; height: 100vh; display: flex; flex-direction: column; }
  #messages { flex: 1; overflow-y: auto; padding: 8px 0; }
  .msg { margin: 8px 0; padding: 10px 14px; border-radius: 12px; max-width: 80%; white-space: pre-wrap; word-break: break-word; }
  .user { background: #1a3a5c; margin-left: auto; }
  .agent { background: #1e1e1e; }
  #input-bar { display: flex; gap: 8px; padding: 8px 0; }
  #input { flex: 1; background: #1e1e1e; border: 1px solid #333; color: #e0e0e0; border-radius: 8px; padding: 10px 14px; font-size: 14px; }
  #input:focus { outline: none; border-color: #3a6ea5; }
  #send { background: #3a6ea5; color: #fff; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-size: 14px; }
  #send:hover { background: #4a7eb5; }
  #send:disabled { background: #333; cursor: default; }
</style>
</head>
<body>
<div id="chat">
  <div id="messages"></div>
  <div id="input-bar">
    <input id="input" placeholder="输入消息..." autocomplete="off">
    <button id="send">发送</button>
  </div>
</div>
<script>
const ws = new WebSocket("ws://" + location.host + "/ws");
const messages = document.getElementById("messages");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

sendBtn.onclick = async () => {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  sendBtn.disabled = true;
  addMsg("user", text);
  ws.send(JSON.stringify({type: "message", content: text}));

  const agentDiv = document.createElement("div");
  agentDiv.className = "msg agent";
  messages.appendChild(agentDiv);
  messages.scrollTop = messages.scrollHeight;

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "delta") {
      agentDiv.textContent += data.content;
      messages.scrollTop = messages.scrollHeight;
    } else if (data.type === "done") {
      sendBtn.disabled = false;
      input.focus();
    } else if (data.type === "proactive") {
      addMsg("agent", data.content);
    }
  };
};

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !sendBtn.disabled) sendBtn.onclick();
});
ws.onopen = () => { addMsg("agent", "ProactiveMind 已连接，输入消息开始对话。"); input.focus(); };
</script>
</body>
</html>"""


def create_app(
    agent: MindLoop,
    connection_manager: SocketHub | None = None,
    health_checker=None,
) -> FastAPI:
    """创建 Web Chat FastAPI 应用。"""

    app = FastAPI(title="ProactiveMind Web Chat")
    cm = connection_manager or SocketHub()
    hc = health_checker

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.get("/health")
    async def health() -> dict:
        if hc is None:
            return {"status": "healthy", "detail": "健康检查未配置"}
        report = hc.check()
        result = hc.to_dict(report)
        if report.status == "unhealthy":
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=503, content=result)
        return result

    @app.get("/stats")
    async def stats() -> dict:
        summary = agent.stats.summary()
        recent = [
            {
                "turn_id": r.turn_id,
                "tool_calls": r.tool_calls,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_ms": r.latency_ms,
                "timestamp": r.timestamp,
            }
            for r in agent.stats.recent(20)
        ]
        return {"summary": summary, "recent": recent}

    @app.get("/sessions")
    async def list_sessions() -> dict:
        return {"sessions": agent.list_sessions()}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        messages = agent.get_session_history(session_id)
        if messages is None:
            return {"error": "会话不存在"}
        return {"session_id": session_id, "messages": messages}

    @app.post("/sessions/switch/{session_id}")
    async def switch_session(session_id: str) -> dict:
        ok = agent.switch_session(session_id)
        if not ok:
            return {"error": "会话不存在"}
        return {"ok": True, "session_id": session_id}

    @app.post("/sessions/reset")
    async def reset_session() -> dict:
        agent.reset_session()
        return {"ok": True, "session_id": agent._session_id}

    @app.get("/sessions/{session_id}/export")
    async def export_session(session_id: str) -> str:
        md = agent.export_session_markdown(session_id)
        if md is None:
            return "会话不存在"
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(md)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await cm.connect(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("type") != "message":
                    continue
                content = data.get("content", "").strip()
                if not content:
                    continue

                async for chunk in agent.run_stream(content):
                    await websocket.send_json({"type": "delta", "content": chunk})
                await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            log.info("WebSocket 客户端断开")
        except Exception as exc:
            log.warning("WebSocket 错误: %s", exc)
        finally:
            cm.disconnect(websocket)

    return app
