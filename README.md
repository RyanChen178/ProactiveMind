# ProactiveMind

带记忆与自驱能力的主动式 AI Agent——不只在被问时回答，更能按需主动推送、空闲自主完成任务。

## 快速开始

需要 Python 3.12+。

```bash
# 安装依赖
pip install -r requirements.txt

# 创建配置文件
cp config.example.toml config.toml
# 编辑 config.toml 填入你的 API Key

# 启动对话
python main.py
```

对话中可使用 `/reset` 新建会话、`/pending` 查看待归档记忆、`/promote` 将未收录的候选记忆追加到 `MEMORY.md`。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 核心特征

- **ReAct 循环** —— 推理 + 行动交替执行，支持多轮工具调用
- **会话持久化** —— 消息写入 SQLite，重启后自动恢复当前会话
- **上下文管理** —— 按 token 预算裁切历史，工具调用与结果始终完整保留
- **流式输出** —— CLI 实时显示模型生成的回复
- **分层提示词** —— 人格、行为规则、工具说明与长期记忆按区块组装
- **持久记忆** —— 对话中保存的事实写入 `MEMORY.md`，跨会话保留
- **自动记忆归档** —— 对话结束后后台提取候选事实，先写入 `PENDING.md`
- **记忆人工提升** —— 使用 `/pending` 查看候选事实，`/promote` 显式追加到长期记忆
- **事件总线** —— emit/fanout/enqueue 三种语义，对话完成事件驱动后台归档
- **工具调用** —— 内置 shell、记忆检索等工具，可扩展
- **OpenAI 兼容** —— 支持任意 OpenAI Chat Completions 兼容端点

## 项目结构

```
proactivemind/
├── main.py              # 入口，CLI 对话 REPL
├── config.example.toml  # 配置模板
├── agent/
│   ├── config.py        # 配置加载
│   ├── provider.py      # LLM 调用（OpenAI 兼容）
│   ├── session.py       # 会话消息历史
│   ├── session_store.py # SQLite 会话存储
│   ├── context.py       # token 估算与历史视图
│   ├── prompt.py        # 分层系统提示词
│   ├── tools.py         # 工具注册 + 内置工具
│   ├── memory.py        # Markdown 文件记忆
│   ├── consolidation.py # 候选长期记忆提取
│   └── loop.py          # Agent ReAct 循环
├── bus/
│   └── __init__.py      # 事件总线（emit/fanout/enqueue）
├── tests/
│   ├── test_bus.py            # 事件总线测试
│   ├── test_loop.py           # ReAct 循环测试
│   └── test_session_store.py  # SQLite 会话存储测试
└── ~/.proactivemind/
    └── workspace/
        ├── sessions.db   # 会话历史
        ├── MEMORY.md     # 已确认的持久记忆
        └── PENDING.md    # 待归档的候选记忆
```

## License

[MIT](./LICENSE)
