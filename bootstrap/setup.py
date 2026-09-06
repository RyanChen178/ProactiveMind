"""交互式 Setup 向导 —— 首次运行生成 config.toml。

启动方式：
  python app.py setup                交互式询问
  python app.py setup --output X.toml 写入指定路径
  python app.py setup --non-interactive --provider ... --api-key ...

设计原则：
- 输入/输出全部通过回调注入（input_fn / print_fn），便于测试
- 生成的 toml 内容用纯字符串拼接，避免引入额外依赖
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]


@dataclass
class WizardAnswers:
    """向导收集到的配置答案。"""

    workspace: str = "~/.proactivemind/workspace"
    main_provider: str = "deepseek"
    main_model: str = "deepseek-v4-flash"
    main_api_key: str = ""
    main_base_url: str = "https://api.deepseek.com/v1"
    enable_fast_runtime: bool = False
    fast_provider: str = "qwen"
    fast_model: str = "qwen-flash"
    fast_api_key: str = ""
    fast_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    enable_telegram: bool = False
    telegram_token: str = ""
    enable_extensions: bool = True
    extensions_dir: str = "extensions"
    overwrite: bool = False
    output_path: str = "config.toml"


def _default_input(prompt: str) -> str:
    """默认输入函数。"""
    return input(prompt)


def _default_print(msg: str) -> None:
    """默认输出函数。"""
    print(msg)


def _ask(
    question: str,
    default: str,
    input_fn: InputFn,
    *,
    required: bool = False,
    is_secret: bool = False,
) -> str:
    """询问用户一个问题，附带默认值。"""
    suffix = f" [{default}]" if default else ""
    prompt = f"{question}{suffix}: "
    if is_secret:
        # 简单处理：直接读，但提示输入时标 secret
        prompt = f"{question}{suffix} (输入隐藏): "

    raw = input_fn(prompt).strip()
    if not raw:
        raw = default
    if required and not raw:
        raise ValueError(f"必填项未填写: {question}")
    return raw


def _ask_yes_no(question: str, default: bool, input_fn: InputFn) -> bool:
    """询问是/否问题。"""
    hint = "Y/n" if default else "y/N"
    raw = input_fn(f"{question} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "是", "1", "true")


def run_setup_interactive(
    *,
    input_fn: InputFn = _default_input,
    print_fn: PrintFn = _default_print,
) -> WizardAnswers:
    """运行交互式 Setup 向导，收集用户答案。"""
    answers = WizardAnswers()

    print_fn("=" * 50)
    print_fn("ProactiveMind 首次配置向导")
    print_fn("=" * 50)
    print_fn("")
    print_fn("直接回车使用默认值；Ctrl+C 中断。")
    print_fn("")

    print_fn("[1/4] 工作区")
    answers.workspace = _ask(
        "工作区路径",
        answers.workspace,
        input_fn,
    )

    print_fn("")
    print_fn("[2/4] 主 LLM 运行时")
    answers.main_provider = _ask("Provider", answers.main_provider, input_fn)
    answers.main_model = _ask("Model", answers.main_model, input_fn)
    answers.main_api_key = _ask(
        "API Key", answers.main_api_key, input_fn, required=True, is_secret=True
    )
    answers.main_base_url = _ask("Base URL", answers.main_base_url, input_fn)

    print_fn("")
    print_fn("[3/4] 轻量 LLM 运行时（可选，用于 memory gate / query rewrite）")
    answers.enable_fast_runtime = _ask_yes_no(
        "启用轻量运行时？", answers.enable_fast_runtime, input_fn
    )
    if answers.enable_fast_runtime:
        answers.fast_provider = _ask("Provider", answers.fast_provider, input_fn)
        answers.fast_model = _ask("Model", answers.fast_model, input_fn)
        answers.fast_api_key = _ask(
            "API Key", answers.fast_api_key, input_fn, required=True, is_secret=True
        )
        answers.fast_base_url = _ask("Base URL", answers.fast_base_url, input_fn)

    print_fn("")
    print_fn("[4/4] 渠道与扩展")
    answers.enable_telegram = _ask_yes_no(
        "启用 Telegram Bot？", answers.enable_telegram, input_fn
    )
    if answers.enable_telegram:
        answers.telegram_token = _ask(
            "Telegram Bot Token",
            answers.telegram_token,
            input_fn,
            required=True,
            is_secret=True,
        )

    answers.enable_extensions = _ask_yes_no(
        "启用扩展系统？", answers.enable_extensions, input_fn
    )
    if answers.enable_extensions:
        answers.extensions_dir = _ask(
            "扩展目录", answers.extensions_dir, input_fn
        )

    answers.output_path = _ask(
        "配置文件输出路径", answers.output_path, input_fn
    )
    answers.overwrite = _ask_yes_no(
        "若目标文件已存在，是否覆盖？", answers.overwrite, input_fn
    )

    return answers


def render_config_toml(answers: WizardAnswers) -> str:
    """根据答案生成完整的 config.toml 文本。"""
    lines: list[str] = []
    lines.append("# ProactiveMind 配置文件（由 setup 向导生成）")
    lines.append("")

    # [llm]
    lines.append("[llm]")
    lines.append(f'main = "main_runtime"')
    if answers.enable_fast_runtime:
        lines.append('fast = "fast_runtime"')
    lines.append("")
    lines.append("[llm.runtimes.main_runtime]")
    lines.append(f'provider = "{answers.main_provider}"')
    lines.append(f'model = "{answers.main_model}"')
    lines.append(f'api_key = "{answers.main_api_key}"')
    lines.append(f'base_url = "{answers.main_base_url}"')
    lines.append('input_modalities = ["text"]')
    lines.append("")
    if answers.enable_fast_runtime:
        lines.append("[llm.runtimes.fast_runtime]")
        lines.append(f'provider = "{answers.fast_provider}"')
        lines.append(f'model = "{answers.fast_model}"')
        lines.append(f'api_key = "{answers.fast_api_key}"')
        lines.append(f'base_url = "{answers.fast_base_url}"')
        lines.append('input_modalities = ["text"]')
        lines.append("")

    # [workspace]
    lines.append("[workspace]")
    lines.append(f'path = "{answers.workspace}"')
    lines.append("")

    # [consolidation]
    lines.append("[consolidation]")
    lines.append("enabled = true")
    lines.append("")

    # [extensions]
    lines.append("[extensions]")
    if answers.enable_extensions:
        lines.append(f'dir = "{answers.extensions_dir}"')
    else:
        lines.append('dir = ""')
    lines.append("")

    # [telegram]
    if answers.enable_telegram:
        lines.append("[telegram]")
        lines.append("enabled = true")
        lines.append(f'token = "{answers.telegram_token}"')
        lines.append("allow_from = []")
        lines.append("")

    # [prompt]
    lines.append("[prompt]")
    lines.append('persona = "你是 ProactiveMind，一个有持久记忆的 AI 助手。"')
    lines.append("rules = [")
    lines.append('  "准确、诚实地回答；不确定时说明不确定性。",')
    lines.append('  "使用工具前先判断是否确有必要。",')
    lines.append('  "重要且长期有效的用户事实可使用 memorize 保存。",')
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


class SetupError(RuntimeError):
    """Setup 流程中的错误。"""

    pass


def save_config(
    content: str,
    path: str | Path,
    *,
    overwrite: bool = False,
    file_exists_fn: Callable[[Path], bool] | None = None,
) -> Path:
    """写入配置文件。

    Args:
        content: toml 内容
        path: 目标路径
        overwrite: 是否覆盖已有文件
        file_exists_fn: 注入的存在性检查（默认 Path.exists）
    """
    target = Path(path)
    exists_check = file_exists_fn or (lambda p: p.exists())
    if exists_check(target) and not overwrite:
        raise SetupError(f"目标文件已存在: {target}（需要 overwrite=True）")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def run_setup_from_args(
    args: dict[str, str | bool] | None = None,
    *,
    input_fn: InputFn = _default_input,
    print_fn: PrintFn = _default_print,
) -> Path:
    """统一入口：支持交互模式和非交互模式。

    Args:
        args: 命令行参数覆盖项。可包含 non_interactive、provider、model、
              api_key、base_url、workspace、output、overwrite 等键
    """
    args = args or {}

    if args.get("non_interactive"):
        answers = _answers_from_args(args)
    else:
        answers = run_setup_interactive(input_fn=input_fn, print_fn=print_fn)

    output = str(args.get("output", answers.output_path))
    overwrite = bool(args.get("overwrite", answers.overwrite))
    content = render_config_toml(answers)
    return save_config(content, output, overwrite=overwrite)


def _answers_from_args(args: dict[str, str | bool]) -> WizardAnswers:
    """从命令行参数构造 WizardAnswers。"""
    answers = WizardAnswers()

    if "workspace" in args:
        answers.workspace = str(args["workspace"])

    if "provider" in args:
        answers.main_provider = str(args["provider"])
    if "model" in args:
        answers.main_model = str(args["model"])
    if "api_key" in args:
        answers.main_api_key = str(args["api_key"])
    if "base_url" in args:
        answers.main_base_url = str(args["base_url"])

    if "enable_fast_runtime" in args:
        answers.enable_fast_runtime = bool(args["enable_fast_runtime"])
    if "fast_provider" in args:
        answers.fast_provider = str(args["fast_provider"])
    if "fast_model" in args:
        answers.fast_model = str(args["fast_model"])
    if "fast_api_key" in args:
        answers.fast_api_key = str(args["fast_api_key"])
    if "fast_base_url" in args:
        answers.fast_base_url = str(args["fast_base_url"])

    if "enable_telegram" in args:
        answers.enable_telegram = bool(args["enable_telegram"])
    if "telegram_token" in args:
        answers.telegram_token = str(args["telegram_token"])

    if "enable_extensions" in args:
        answers.enable_extensions = bool(args["enable_extensions"])
    if "extensions_dir" in args:
        answers.extensions_dir = str(args["extensions_dir"])

    if "output" in args:
        answers.output_path = str(args["output"])
    if "overwrite" in args:
        answers.overwrite = bool(args["overwrite"])

    return answers