"""性能评测脚本 —— 评估 ProactiveMind 各模块性能。

用法：
  python benchmark/eval_runner.py [场景名...]

场景：
  - memory_search    TF-IDF 向量检索吞吐量
  - context_compact 上下文压缩速度
  - session_list    会话列表查询延迟
  - presence        用户活跃记录吞吐
  - all（默认）

输出为 JSON 报告 + stdout 表格。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class BenchmarkResult:
    """单个评测结果。"""

    scenario: str
    iterations: int
    total_seconds: float
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    throughput: float  # ops/sec


def _measure(
    scenario: str,
    iterations: int,
    func: Callable[[], None],
) -> BenchmarkResult:
    """运行 iterations 次 func，统计延迟分布。"""
    samples_ms: list[float] = []
    start_total = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        samples_ms.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - start_total

    samples_ms.sort()
    p50 = samples_ms[len(samples_ms) // 2]
    p95 = samples_ms[int(len(samples_ms) * 0.95)] if samples_ms else 0.0

    return BenchmarkResult(
        scenario=scenario,
        iterations=iterations,
        total_seconds=total,
        avg_ms=statistics.mean(samples_ms) if samples_ms else 0.0,
        min_ms=min(samples_ms) if samples_ms else 0.0,
        max_ms=max(samples_ms) if samples_ms else 0.0,
        p50_ms=p50,
        p95_ms=p95,
        throughput=iterations / total if total > 0 else 0.0,
    )


def bench_memory_search(iterations: int) -> BenchmarkResult:
    """TF-IDF 向量检索吞吐。"""
    from mind.vector_store import VectorStore

    store = VectorStore()
    # 准备 200 条中文语料
    corpus = [
            f"用户喜欢第{i}种编程语言和工作流程{i}"
            for i in range(200)
        ]
    for text in corpus:
        store.add(text)
    queries = ["编程语言", "工作流程", "Python", "中文检索", "用户偏好"]

    def _one_query() -> None:
        for q in queries:
            store.search(q, top_k=5)

    return _measure("memory_search", iterations, _one_query)


def bench_context_compact(iterations: int) -> BenchmarkResult:
    """上下文压缩速度（不调用 LLM，只测量消息预算计算）。"""
    from mind.compaction import (
        ContextCompactor,
        estimate_message_tokens,
    )

    # 不需要真实 provider，只用 token 估算测量裁剪开销
    history = [
        {"role": "user", "content": f"问题 {i}：项目细节描述..." * 20}
        for i in range(60)
    ]
    history_with_assistant: list[dict] = []
    for i, msg in enumerate(history):
        history_with_assistant.append(msg)
        if i % 2 == 0:
            history_with_assistant.append({
                "role": "assistant",
                "content": f"回答 {i}：项目细节描述..." * 20,
            })

    keep_recent = 2000

    def _compact() -> None:
        # 模拟压缩的 token 估算 + 切片逻辑（不调 LLM）
        total = sum(estimate_message_tokens(m) for m in history_with_assistant)
        if total <= keep_recent:
            return
        kept: list[dict] = []
        used = 0
        for msg in reversed(history_with_assistant):
            t = estimate_message_tokens(msg)
            if used + t > keep_recent and kept:
                break
            kept.append(msg)
            used += t
        kept.reverse()

    return _measure("context_compact", iterations, _compact)


def bench_session_list(iterations: int) -> BenchmarkResult:
    """会话列表查询延迟。"""
    from mind.session_store import SessionStore

    # 用绝对路径避免 Windows 临时目录的清理时序问题
    db_path = Path(tempfile.gettempdir()) / "proactivemind_bench_sessions.db"
    if db_path.exists():
        db_path.unlink()

    store = SessionStore(db_path)
    try:
        # 预先创建 50 个会话
        for i in range(50):
            store.get_or_create_active_session()
            store.append_message(
                store.get_or_create_active_session(),
                {"role": "user", "content": f"问题 {i}"},
            )

        def _list() -> None:
            store.list_sessions()

        return _measure("session_list", iterations, _list)
    finally:
        store.close()
        if db_path.exists():
            db_path.unlink()


def bench_presence(iterations: int) -> BenchmarkResult:
    """PresenceStore 记录延迟。"""
    from datetime import datetime, timezone
    from initiative.presence import PresenceStore

    with tempfile.TemporaryDirectory() as temp_dir:
        store = PresenceStore(Path(temp_dir) / "presence.db")
        ts = datetime.now(timezone.utc)

        def _record() -> None:
            store.record_user_message(ts)

        result = _measure("presence", iterations, _record)
        store.close()
        return result


SCENARIOS: dict[str, Callable[[int], BenchmarkResult]] = {
    "memory_search": bench_memory_search,
    "context_compact": bench_context_compact,
    "session_list": bench_session_list,
    "presence": bench_presence,
}


def print_table(results: list[BenchmarkResult]) -> None:
    """打印结果表格。"""
    headers = ["场景", "迭代", "avg ms", "p50 ms", "p95 ms", "max ms", "ops/s"]
    rows = [
        [
            r.scenario,
            str(r.iterations),
            f"{r.avg_ms:.2f}",
            f"{r.p50_ms:.2f}",
            f"{r.p95_ms:.2f}",
            f"{r.max_ms:.2f}",
            f"{r.throughput:.1f}",
        ]
        for r in results
    ]

    widths = [
        max(len(row[i]) for row in [headers] + rows)
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["─" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def run_scenarios(
    names: list[str],
    iterations: int,
    output_path: Path | None = None,
) -> list[BenchmarkResult]:
    """运行指定场景。"""
    if not names or "all" in names:
        names = list(SCENARIOS.keys())

    results: list[BenchmarkResult] = []
    for name in names:
        if name not in SCENARIOS:
            print(f"未知场景: {name}", file=sys.stderr)
            continue
        print(f"运行场景: {name} ...", file=sys.stderr)
        try:
            result = SCENARIOS[name](iterations)
        except Exception as exc:
            print(f"  失败: {exc}", file=sys.stderr)
            continue
        results.append(result)
        print(f"  完成: avg={result.avg_ms:.2f}ms p95={result.p95_ms:.2f}ms", file=sys.stderr)

    print_table(results)

    if output_path is not None:
        output_path.write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n报告已写入: {output_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="ProactiveMind 性能评测")
    parser.add_argument(
        "scenarios",
        nargs="*",
        help=f"场景名（{', '.join(SCENARIOS.keys())}）默认全部",
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=200,
        help="每个场景的迭代次数（默认 200）",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="JSON 报告输出路径",
    )
    args = parser.parse_args()

    run_scenarios(args.scenarios, args.iterations, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())