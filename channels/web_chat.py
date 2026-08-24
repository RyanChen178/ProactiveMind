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

from agent.loop import AgentLoop

log = logging.getLogger(__name__)


class ConnectionManager:
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
    agent: AgentLoop,
    connection_manager: ConnectionManager | None = None,
) -> FastAPI:
    """创建 Web Chat FastAPI 应用。"""

    app = FastAPI(title="ProactiveMind Web Chat")
    cm = connection_manager or ConnectionManager()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

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
