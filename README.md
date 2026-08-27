# ProactiveMind

带记忆与自驱能力的主动式 AI Agent——不只在被问时回答，更能按需主动推送、空闲自主完成任务。

## 快速开始

需要 Python 3.12+。

```bash
# 安装依赖
pip install -r requirements.txt

# 创建配置文件
cp config.sample.toml config.toml
# 编辑 config.toml 填入你的 API Key

# 启动对话
python app.py

# 启动 Web Chat
pip install -e ".[web]"
python app.py web
# 浏览器访问 http://127.0.0.1:6322
```

对话中可使用 `/reset` 新建会话、`/pending` 查看待归档记忆、`/promote` 将未收录的候选记忆追加到 `MEMORY.md`。

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

- **ReAct 循环** —— 推理 + 行动交替执行，支持多轮工具调用
- **会话持久化** —— 消息写入 SQLite，重启后自动恢复当前会话
- **上下文管理** —— 按 token 预算裁切历史，工具调用与结果始终完整保留
- **流式输出** —— CLI 实时显示模型生成的回复
- **分层提示词** —— 人格、行为规则、工具说明与长期记忆按区块组装
- **持久记忆** —— 对话中保存的事实写入 `MEMORY.md`，跨会话保留
- **语义检索** —— TF-IDF + 余弦相似度的向量记忆搜索，recall 工具优先语义匹配
- **自动记忆归档** —— 对话结束后后台提取候选事实，先写入 `PENDING.md`
- **记忆人工提升** —— 使用 `/pending` 查看候选事实，`/promote` 显式追加到长期记忆
- **事件枢纽** —— emit/fanout/enqueue 三种语义，对话完成事件驱动后台归档
- **主动推送** —— 电量模型自适应轮询，用户空闲越久轮询越频繁；Wander 结果可推送到 Web Chat 客户端
- **Wander 空闲任务** —— 无内容可推时自主执行后台 playbook
- **Web Chat** —— FastAPI + WebSocket 浏览器对话界面，REST API 管理会话（列出/切换/导出）
- **扩展系统** —— 声明式工具注册，extensions/ 目录自动发现，MindLoop 启动时自动加载
- **工具调用** —— 内置 shell、记忆检索等工具，可扩展
- **工具权限** —— shell 命令安全审查，拦截 rm -rf、mkfs、dd 等危险操作
- **Turn 指标** —— 记录每轮对话的 token 用量、延迟、工具调用，`/stats` 端点可查
- **健康检查** —— `/health` 端点检查记忆、会话、心跳等核心组件状态
- **配置校验** —— 启动时校验配置合法性，提前发现问题
- **OpenAI 兼容** —— 支持任意 OpenAI Chat Completions 兼容端点

## 项目结构

```
proactivemind/
├── app.py               # 入口（CLI / Web 模式）
├── config.sample.toml   # 配置模板
├── mind/
│   ├── config.py        # 配置加载
│   ├── provider.py      # LLM 调用（OpenAI 兼容）
│   ├── session.py       # 会话消息历史
│   ├── session_store.py # SQLite 会话存储
│   ├── context.py       # token 估算与历史视图
│   ├── prompt.py        # 分层系统提示词
│   ├── tools.py         # 工具注册 + 内置工具
│   ├── permission.py    # 工具权限（shell 命令安全审查）
│   ├── stats.py         # Turn 指标收集
│   ├── health.py        # 健康检查
│   ├── vector_store.py  # 向量记忆检索（TF-IDF + 余弦相似度）
│   ├── memory.py        # Markdown 文件记忆
│   ├── consolidation.py # 候选长期记忆提取
│   └── loop.py          # MindLoop ReAct 循环
├── events/
│   └── __init__.py      # 事件枢纽（emit/fanout/enqueue）
├── gateways/
│   └── web_chat.py      # FastAPI + WebSocket Web Chat
├── extensions/
│   ├── __init__.py      # 扩展系统公共接口
│   ├── manager.py       # ExtensionManager（发现 + 加载 + 注册）
│   └── notes.py         # 示例扩展（take_note / list_notes）
├── initiative/
│   ├── energy.py        # 电量模型（三段衰减 + 自适应间隔）
│   ├── presence.py      # 用户活跃心跳追踪
│   ├── drift.py         # Wander 空闲任务（扫描执行 PLAYBOOK.md）
│   └── loop.py          # 主动推送定时循环
├── playbooks/
│   └── audit-memory/
│       └── PLAYBOOK.md  # 后台任务指南示例
├── tests/
│   ├── test_events.py         # 事件枢纽测试
│   ├── test_energy.py         # 电量模型测试
│   ├── test_initiative.py     # 主动推送循环测试
│   ├── test_wander.py         # Wander 空闲任务测试
│   ├── test_web_chat.py       # Web Chat 网关测试
│   ├── test_socket_hub.py     # WebSocket 连接管理测试
│   ├── test_extensions.py     # 扩展系统测试
│   ├── test_extension_integration.py  # 扩展集成测试
│   ├── test_permission.py     # 工具权限测试
│   ├── test_mind_loop.py      # ReAct 循环测试
│   ├── test_stats.py          # Turn 指标测试
│   ├── test_health.py         # 健康检查与配置校验测试
│   ├── test_vector_store.py   # 向量记忆检索测试
│   ├── test_session_api.py    # 会话管理 REST API 测试
│   └── test_session_store.py  # SQLite 会话存储测试
└── ~/.proactivemind/
    └── workspace/
        ├── sessions.db    # 会话历史
        ├── presence.db    # 用户活跃心跳
        ├── MEMORY.md      # 已确认的持久记忆
        └── PENDING.md     # 待归档的候选记忆
```

## License

[MIT](./LICENSE)
