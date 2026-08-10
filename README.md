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

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 核心特征

- **ReAct 循环** —— 推理 + 行动交替执行，支持多轮工具调用
- **会话持久化** —— 消息写入 SQLite，重启后自动恢复当前会话
- **持久记忆** —— 对话中保存的事实写入 `MEMORY.md`，跨会话保留
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
│   ├── tools.py         # 工具注册 + 内置工具
│   ├── memory.py        # Markdown 文件记忆
│   └── loop.py          # Agent ReAct 循环
├── tests/
│   ├── test_loop.py          # ReAct 循环测试
│   └── test_session_store.py # SQLite 会话存储测试
└── ~/.proactivemind/
    └── workspace/
        ├── sessions.db   # 会话历史
        └── MEMORY.md    # 持久记忆
```

## License

[MIT](./LICENSE)
