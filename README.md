# ProactiveMind

带记忆与自驱能力的主动式 AI Agent——不只在被问时回答，更能按需主动推送、空闲自主完成任务。

## 快速开始

需要 Python 3.12+。

```bash
# 安装依赖
pip install -r requirements.txt

# 交互式配置向导（推荐首次使用）
python app.py setup

# 或手动创建配置文件
cp config.sample.toml config.toml
# 编辑 config.toml 填入你的 API Key

# 启动对话
python app.py

# 启动 Web Chat（http://127.0.0.1:6322）
python app.py web
```

## 启动模式

| 命令 | 说明 |
|---|---|
| `python app.py` | CLI 对话 REPL（`/help` 查看全部命令） |
| `python app.py web` | Web Chat，浏览器访问 http://127.0.0.1:6322 |
| `python app.py telegram` | Telegram Bot 长轮询渠道 |
| `python app.py dashboard` | 只读调试面板 http://127.0.0.1:6323（状态/记忆/工具/MCP） |
| `python app.py control` | Agent 控制协议服务端 127.0.0.1:6324（JSON-RPC over TCP） |
| `python app.py setup` | 交互式配置向导，支持 `--non-interactive` 参数 |
| `python app.py supervise <web\|telegram>` | Supervisor 托管 gateway 子进程，崩溃自动重启 |

对话中可用的部分命令：`/help`、`/skills`、`/skill <name>`、`/search <query>`、`/notes [tag]`、`/tools`、`/mcp`、`/optimize`、`/stats`。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

CI 会自动在 Python 3.12 / 3.13 上运行测试和 lint。

## Docker 部署

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

默认暴露 `6322` 端口，配置文件从 `config.toml` 挂载，扩展目录从 `extensions/` 挂载。

## 核心特征

### 对话内核
- **ReAct 循环** —— 推理 + 行动交替执行，支持多轮工具调用
- **Turn 中断** —— `interrupt_current()` 在流式输出的任意检查点打断本轮
- **多模型运行时** —— `[llm.runtimes.xxx]` 命名运行时，主/快速双模型分工
- **模型感知上下文压缩** —— 按 context_window 自动压缩历史，checkpoint 记录压缩点
- **轻量模型辅助** —— 用快速模型做 memory gate / query rewrite / 事实去重
- **流式输出** —— CLI / WebSocket / Telegram / 控制协议全渠道流式
- **分层提示词** —— 人格、行为规则、工具说明与长期记忆按区块组装

### 记忆系统
- **持久记忆** —— 事实写入 `MEMORY.md` 跨会话保留，`PENDING.md` 缓存候选
- **自动归档** —— Optimizer 定时把候选记忆质量过滤后并入长期记忆（`/optimize` 手动触发）
- **语义检索** —— TF-IDF 与可选 embedding 后端（本地/HTTP 双实现，SQLite 向量缓存）
- **三路数据源** —— 主动推送拉取 alert / content / context 三路内容

### 主动能力
- **主动推送** —— 电量模型三段衰减自适应轮询；限频器防打扰；Turn 执行期间自动避让
- **Wander 空闲任务** —— 空闲时自主执行后台 playbook，也可 `/skill` 手动触发
- **Self.md 自我模型** —— Agent 维护自身偏好与目标的认知文件

### 工具与扩展
- **内置工具** —— get_time / shell / memorize / recall / web_fetch
- **web_fetch** —— URL 抓取 + HTML 转文本 + SSRF 防护 + 重试退避
- **MCP 客户端** —— 接入任意 Model Context Protocol server，工具自动注册为 `mcp_<server>__<tool>`
- **扩展系统** —— 声明式注册 + 自动发现 + 6 阶段生命周期钩子 + 工具拦截器 + 热重载
- **工具权限** —— shell 命令安全审查，拦截 rm -rf、mkfs、dd 等危险操作

### 接入层
- **Web Chat** —— FastAPI + WebSocket，会话管理 REST API
- **Telegram** —— Bot API 长轮询，白名单 + 消息分片 + 主动推送
- **Dashboard** —— 只读调试面板（状态/记忆/统计/工具/MCP server）
- **控制协议** —— JSON-RPC 2.0 over TCP，配套 Python SDK（`sdk/python/`）
- **Supervisor** —— 进程守护，指数退避重启，PID 锁防多开

### 基础设施
- **结构化日志** —— JSON 格式输出，trace_id/span_id 贯穿整轮对话
- **HTTP 重试** —— 统一重试策略（指数退避 + 抖动 + 总超时预算）
- **Turn 指标** —— token 用量、延迟、工具调用统计
- **健康检查** —— `/health` 端点 + 配置校验
- **性能评测** —— `benchmark/eval_runner.py` 基准脚本

## SDK

```python
from proactivemind_sdk import AsyncProactiveMind

client = await AsyncProactiveMind.connect("127.0.0.1:6324")
listing = await client.session_list()
sub = await client.turn_start(session_id, "你好")
async for event in sub.events():
    ...  # delta / turn_done 事件流
```

## 项目结构

```
proactivemind/
├── app.py               # 入口（cli / web / telegram / dashboard / control / setup / supervise）
├── cli_commands.py      # REPL 命令路由
├── config.sample.toml   # 配置模板
├── core/
│   ├── network.py       # HTTP 重试 + 指数退避
│   └── diagnostics.py   # 结构化 JSON 日志 + trace 传播
├── mind/
│   ├── loop.py          # MindLoop ReAct 循环（中断/trace/MCP/Optimizer 集成）
│   ├── provider.py      # LLM 调用（多运行时）
│   ├── compaction.py    # 模型感知上下文压缩
│   ├── lightweight_assistant.py  # 轻量模型辅助
│   ├── tools.py         # 工具注册 + 内置工具
│   ├── web_fetch.py     # URL 抓取（SSRF 防护 + HTML 转文本）
│   ├── memory.py        # Markdown 文件记忆 + 语义检索
│   ├── embeddings/      # embedding 后端 + 向量缓存
│   ├── mcp/             # MCP 客户端（JSON-RPC over stdio）
│   ├── optimizer.py     # PENDING 定时归档
│   ├── vector_store.py  # TF-IDF 向量检索
│   ├── extensions/      # 生命周期钩子/事件总线/拦截器/热重载
│   └── ...
├── events/              # 事件枢纽（emit/fanout/enqueue）
├── gateways/
│   ├── web_chat.py      # FastAPI + WebSocket Web Chat
│   ├── telegram_bot.py  # Telegram Bot 渠道
│   ├── dashboard.py     # 调试面板
│   └── control_server.py# 控制协议服务端
├── extensions/
│   ├── manager.py       # ExtensionManager
│   └── notes.py         # 持久化笔记扩展（示例）
├── initiative/
│   ├── loop.py          # 主动推送循环（三路数据源集成）
│   ├── energy.py        # 电量模型
│   ├── rate_limiter.py  # 推送限频（滑动窗口 + 令牌桶）
│   ├── data_sources.py  # alert / content / context 三路数据源
│   ├── self_model.py    # Self.md 自我模型
│   └── ...
├── bootstrap/
│   ├── setup.py         # 配置向导
│   └── supervisor.py    # 进程守护
├── sdk/python/          # Python SDK（JSON-RPC 客户端）
├── benchmark/           # 性能评测
├── playbooks/           # 后台任务指南
└── tests/               # 540+ 单元测试
```

## License

[MIT](./LICENSE)
