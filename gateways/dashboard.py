"""Dashboard 调试入口 —— 只读的内部状态视图。

启动方式：
  python app.py dashboard       —— http://127.0.0.1:6323

端点：
  GET  /                      暗色 HTML 概览页
  GET  /api/status            Agent 状态 JSON
  GET  /api/presence          用户活跃状态
  GET  /api/memory            PENDING / MEMORY 内容
  GET  /api/stats             Turn 统计
  GET  /api/sessions          会话列表
  POST /api/promote           提升待归档记忆
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from initiative.presence import PresenceStore
from mind.loop import MindLoop

log = logging.getLogger(__name__)


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ProactiveMind Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1419; color: #e6e6e6; line-height: 1.5; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
  header { display: flex; align-items: center; gap: 12px; margin-bottom: 24px; }
  h1 { font-size: 22px; font-weight: 600; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 12px;
           background: #1f6feb33; color: #79b8ff; border: 1px solid #1f6feb55; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px; }
  .card h2 { font-size: 14px; color: #8b949e; margin-bottom: 12px;
             text-transform: uppercase; letter-spacing: 0.05em; }
  .stat { display: flex; justify-content: space-between; padding: 6px 0;
          border-bottom: 1px solid #21262d; font-size: 14px; }
  .stat:last-child { border-bottom: none; }
  .stat .label { color: #8b949e; }
  .stat .value { color: #e6e6e6; font-weight: 500; }
  pre { background: #0d1117; padding: 12px; border-radius: 6px; overflow-x: auto;
        font-size: 12px; max-height: 280px; }
  ul { list-style: none; }
  li { padding: 6px 0; font-size: 14px; border-bottom: 1px solid #21262d; }
  li:last-child { border-bottom: none; }
  .empty { color: #6e7681; font-style: italic; }
  button { background: #238636; color: #fff; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; font-size: 14px; }
  button:hover { background: #2ea043; }
  button:disabled { background: #21262d; color: #6e7681; cursor: default; }
  .toolbar { margin: 12px 0; display: flex; gap: 8px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>ProactiveMind Dashboard</h1>
    <span class="badge">debug</span>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Agent 状态</h2>
      <div id="status">加载中...</div>
    </div>
    <div class="card">
      <h2>用户活跃</h2>
      <div id="presence">加载中...</div>
    </div>
    <div class="card">
      <h2>Turn 统计</h2>
      <div id="stats">加载中...</div>
    </div>
    <div class="card">
      <h2>会话列表</h2>
      <div id="sessions">加载中...</div>
    </div>
    <div class="card" style="grid-column: span 2;">
      <h2>待归档记忆 (PENDING)</h2>
      <div class="toolbar">
        <button onclick="promote()">提升为长期记忆</button>
      </div>
      <ul id="pending"></ul>
    </div>
    <div class="card" style="grid-column: span 2;">
      <h2>长期记忆 (MEMORY)</h2>
      <ul id="memory"></ul>
    </div>
  </div>
</div>
<script>
async function fetch_json(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + ' ' + r.status);
  return r.json();
}
function kv(label, value) {
  return `<div class="stat"><span class="label">${label}</span><span class="value">${value}</span></div>`;
}
async function refresh() {
  try {
    const status = await fetch_json('/api/status');
    document.getElementById('status').innerHTML =
      kv('会话 ID', status.session_id || '—') +
      kv('当前轮用户消息数', status.user_message_count) +
      kv('最近消息', status.last_user_message || '—');

    const presence = await fetch_json('/api/presence');
    document.getElementById('presence').innerHTML =
      kv('最后用户活跃', presence.last_user_at || '从未') +
      kv('最后主动推送', presence.last_proactive_at || '从未');

    const stats = await fetch_json('/api/stats');
    document.getElementById('stats').innerHTML = stats.empty
      ? '<div class="empty">暂无统计</div>'
      : kv('总轮次', stats.total_turns) +
        kv('总 token', stats.total_tokens) +
        kv('平均延迟', stats.avg_latency_ms.toFixed(0) + ' ms');

    const sessions = await fetch_json('/api/sessions');
    document.getElementById('sessions').innerHTML = sessions.sessions.length === 0
      ? '<div class="empty">暂无会话</div>'
      : sessions.sessions.map(s =>
          `<div class="stat"><span class="label">${s.session_id.slice(0, 12)}…</span>` +
          `<span class="value">${s.message_count} 条</span></div>`
        ).join('');

    const mem = await fetch_json('/api/memory');
    document.getElementById('pending').innerHTML = mem.pending.length === 0
      ? '<li class="empty">没有待归档记忆</li>'
      : mem.pending.map(f => `<li>${escape(f)}</li>`).join('');
    document.getElementById('memory').innerHTML = mem.memory.length === 0
      ? '<li class="empty">尚无长期记忆</li>'
      : mem.memory.map(f => `<li>${escape(f)}</li>`).join('');
  } catch (e) {
    console.error(e);
  }
}
function escape(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  })[c]);
}
async function promote() {
  const btn = event.target;
  btn.disabled = true;
  try {
    const r = await fetch('/api/promote', { method: 'POST' });
    const data = await r.json();
    alert('已提升 ' + data.promoted + ' 条记忆');
    refresh();
  } finally {
    btn.disabled = false;
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def _format_ts(value: str | None) -> str:
    """格式化 ISO 时间戳。"""
    if not value:
        return "从未"
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def create_dashboard_app(
    agent: MindLoop,
    presence: PresenceStore,
    workspace: Path,
) -> FastAPI:
    """构造 Dashboard FastAPI 应用。"""

    app = FastAPI(title="ProactiveMind Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return HTML_PAGE

    @app.get("/api/status")
    async def status() -> dict:
        """Agent 内部状态。"""
        session = agent._session
        messages = list(session.messages)
        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else None
        return {
            "session_id": agent._session_id,
            "user_message_count": len(user_msgs),
            "last_user_message": last_user,
        }

    @app.get("/api/presence")
    async def presence_status() -> dict:
        """用户活跃状态。"""
        last_user = presence.get_last_user_at()
        last_proactive = presence.get_last_proactive_at()
        return {
            "last_user_at": last_user.isoformat() if last_user else None,
            "last_proactive_at": (
                last_proactive.isoformat() if last_proactive else None
            ),
            "now": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/stats")
    async def stats() -> dict:
        """Turn 统计摘要。"""
        summary = agent._stats.summary()
        if summary.get("total_turns", 0) == 0:
            return {"empty": True, **summary}
        return summary

    @app.get("/api/sessions")
    async def sessions() -> dict:
        """会话列表。"""
        items = agent._session_store.list_sessions()
        return {"sessions": items}

    @app.get("/api/memory")
    async def memory() -> dict:
        """PENDING 与 MEMORY 内容。"""
        return {
            "pending": agent._memory.unpromoted_pending(),
            "memory": _read_memory_facts(workspace / "MEMORY.md"),
        }

    @app.post("/api/promote")
    async def promote() -> dict:
        """手动提升待归档记忆。"""
        promoted = agent._memory.promote_pending()
        return {"promoted": len(promoted), "facts": promoted}

    return app


def _read_memory_facts(path: Path) -> list[str]:
    """读取 MEMORY.md 中所有事实行。"""
    if not path.exists():
        return []
    facts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- "):
            facts.append(line[2:])
    return facts