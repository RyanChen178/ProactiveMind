"""Supervisor 进程管理测试。"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from bootstrap.supervisor import (
    CHILD_ENV_VAR,
    PidFileLock,
    Supervisor,
    SupervisorConfig,
    SupervisorError,
    is_child_process,
    supervise,
)


class SupervisorConfigTest(unittest.TestCase):
    """配置参数校验。"""

    def test_invalid_gateway_mode(self) -> None:
        with self.assertRaises(ValueError):
            SupervisorConfig(workspace=Path("/tmp"), gateway_mode="invalid")

    def test_negative_max_restarts(self) -> None:
        with self.assertRaises(ValueError):
            SupervisorConfig(
                workspace=Path("/tmp"), gateway_mode="web", max_restarts=-1
            )

    def test_non_positive_backoff_base(self) -> None:
        with self.assertRaises(ValueError):
            SupervisorConfig(
                workspace=Path("/tmp"), gateway_mode="web", backoff_base=0
            )


class PidFileLockTest(unittest.TestCase):
    """PID 文件锁测试。"""

    def test_acquire_succeeds_when_no_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock = PidFileLock(Path(temp_dir) / "test.lock")
            try:
                self.assertTrue(lock.acquire())
                self.assertTrue(lock.path.exists())
                self.assertEqual(lock._read_pid(), os.getpid())
            finally:
                lock.release()

    def test_acquire_fails_when_existing_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "test.lock"
            # 先写入一个"活着的" PID（当前进程）
            lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

            lock = PidFileLock(lock_path)
            self.assertFalse(lock.acquire())

    def test_acquire_succeeds_when_existing_dead(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "test.lock"
            # 写入一个不存在的 PID
            lock_path.write_text("999999999\n", encoding="utf-8")

            lock = PidFileLock(lock_path)
            try:
                self.assertTrue(lock.acquire())
            finally:
                lock.release()

    def test_release_only_clears_own_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "test.lock"
            # 模拟别的进程的锁
            lock_path.write_text("999999999\n", encoding="utf-8")

            lock = PidFileLock(lock_path)
            lock.release()
            # 文件应仍在
            self.assertTrue(lock_path.exists())

    def test_release_idempotent_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock = PidFileLock(Path(temp_dir) / "test.lock")
            lock.release()  # 不存在也不报错
            self.assertFalse(lock.path.exists())


class SupervisorRunTest(unittest.IsolatedAsyncioTestCase):
    """Supervisor 主循环行为测试（注入 mock spawn/sleep/now）。"""

    def _make_supervisor(
        self,
        temp_dir: str,
        *,
        exit_sequence: list[int] | None = None,
        max_restarts: int = 3,
    ) -> Supervisor:
        cfg = SupervisorConfig(
            workspace=Path(temp_dir),
            gateway_mode="web",
            max_restarts=max_restarts,
            backoff_base=0.01,
            backoff_max=0.05,
            shutdown_timeout=0.5,
        )

        # 构造一个可控的 spawn 函数：按序列返回 Popen-like 对象
        exit_sequence = exit_sequence or []
        pops: list[MagicMock] = []

        def spawn_fn(_cfg: SupervisorConfig) -> MagicMock:
            if not exit_sequence:
                exit_code = 0  # 默认干净退出
            else:
                exit_code = exit_sequence.pop(0)
            popen = MagicMock()
            popen.pid = 1000 + len(pops)
            popen.wait = MagicMock(return_value=exit_code)
            popen.poll = MagicMock(return_value=exit_code)
            popen.send_signal = MagicMock()
            popen.kill = MagicMock()
            pops.append(popen)
            return popen

        sleeps: list[float] = []

        def sleep_fn(seconds: float) -> None:
            sleeps.append(seconds)

        now_values = [1000.0]

        def now_fn() -> float:
            return now_values[0]

        sup = Supervisor(
            cfg,
            spawn_fn=spawn_fn,
            sleep_fn=sleep_fn,
            now_fn=now_fn,
        )
        sup._test_pops = pops  # type: ignore[attr-defined]
        sup._test_sleeps = sleeps  # type: ignore[attr-defined]
        return sup

    async def test_clean_exit_stops_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sup = self._make_supervisor(temp_dir, exit_sequence=[0])
            sup.acquire_lock()
            try:
                exit_code = sup.run()
                self.assertEqual(exit_code, 0)
                # 仅 spawn 一次
                self.assertEqual(len(sup._test_pops), 1)  # type: ignore[attr-defined]
                self.assertEqual(sup._child.exit_code, 0)
            finally:
                sup.release_lock()

    async def test_restart_after_unexpected_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 第一次异常退出（code=1），第二次干净退出
            sup = self._make_supervisor(
                temp_dir, exit_sequence=[1, 0], max_restarts=3
            )
            sup.acquire_lock()
            try:
                exit_code = sup.run()
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(sup._test_pops), 2)  # type: ignore[attr-defined]
                self.assertEqual(sup._child.restart_count, 1)
                # 应有一次退避 sleep
                self.assertGreater(len(sup._test_sleeps), 0)  # type: ignore[attr-defined]
            finally:
                sup.release_lock()

    async def test_exits_after_max_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 5 次异常退出，超过 max_restarts=2
            sup = self._make_supervisor(
                temp_dir, exit_sequence=[1, 1, 1, 1, 1], max_restarts=2
            )
            sup.acquire_lock()
            try:
                exit_code = sup.run()
                self.assertEqual(exit_code, 1)
                # 启动 1 + 重启 2 = 3 次
                self.assertEqual(len(sup._test_pops), 3)  # type: ignore[attr-defined]
                self.assertEqual(sup._child.restart_count, 3)
            finally:
                sup.release_lock()

    async def test_stop_request_skips_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = SupervisorConfig(
                workspace=Path(temp_dir),
                gateway_mode="web",
                max_restarts=5,
                backoff_base=0.01,
                backoff_max=0.05,
            )

            def spawn_fn(_c: SupervisorConfig) -> MagicMock:
                popen = MagicMock()
                popen.pid = 999
                popen.wait = MagicMock(return_value=0)
                popen.poll = MagicMock(return_value=0)
                return popen

            sup = Supervisor(cfg, spawn_fn=spawn_fn)
            sup.acquire_lock()
            try:
                # 启动后立即请求停止
                original_spawn = sup._spawn

                def delayed_stop():
                    sup.request_stop()
                    return original_spawn()

                sup._spawn = delayed_stop  # type: ignore[assignment]
                exit_code = sup.run()
                self.assertEqual(exit_code, 0)
            finally:
                sup.release_lock()


class SupervisorBackoffTest(unittest.TestCase):
    """退避时间计算。"""

    def test_exponential_growth(self) -> None:
        cfg = SupervisorConfig(
            workspace=Path("/tmp"),
            gateway_mode="web",
            backoff_base=1.0,
            backoff_max=60.0,
        )
        sup = Supervisor(cfg)

        # 第 0 次重启：1.0s
        # 第 1 次：2.0s
        # 第 2 次：4.0s
        self.assertEqual(sup._compute_backoff(0), 1.0)
        self.assertEqual(sup._compute_backoff(1), 2.0)
        self.assertEqual(sup._compute_backoff(2), 4.0)

    def test_clamped_to_max(self) -> None:
        cfg = SupervisorConfig(
            workspace=Path("/tmp"),
            gateway_mode="web",
            backoff_base=1.0,
            backoff_max=10.0,
        )
        sup = Supervisor(cfg)

        # 2^10 = 1024 应被截断到 10.0
        self.assertEqual(sup._compute_backoff(10), 10.0)
        self.assertEqual(sup._compute_backoff(20), 10.0)


class ChildDetectionTest(unittest.TestCase):
    """子进程标识检测。"""

    def test_returns_false_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(CHILD_ENV_VAR, None)
            self.assertFalse(is_child_process())

    def test_returns_true_when_env_set(self) -> None:
        with patch.dict(os.environ, {CHILD_ENV_VAR: "1"}):
            self.assertTrue(is_child_process())


class AcquireLockTest(unittest.IsolatedAsyncioTestCase):
    """Supervisor 锁获取失败处理。"""

    async def test_raises_when_already_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = SupervisorConfig(
                workspace=Path(temp_dir), gateway_mode="web"
            )
            sup1 = Supervisor(cfg)
            sup1.acquire_lock()
            try:
                # 第二个 supervisor 应获取锁失败
                sup2 = Supervisor(cfg)
                with self.assertRaises(SupervisorError):
                    sup2.acquire_lock()
            finally:
                sup1.release_lock()


class TerminateChildTest(unittest.IsolatedAsyncioTestCase):
    """子进程终止流程。"""

    async def test_terminate_calls_kill_when_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = SupervisorConfig(
                workspace=Path(temp_dir),
                gateway_mode="web",
                shutdown_timeout=0.1,
            )
            sup = Supervisor(cfg)

            # 构造一个永远不退出的子进程
            popen = MagicMock()
            popen.pid = 1234
            popen.poll = MagicMock(return_value=None)  # 仍然活着
            popen.send_signal = MagicMock()
            popen.kill = MagicMock()
            sup._popen_handle = popen
            sup._child.pid = 1234

            # 把 _is_child_alive 强制返回 True（模拟子进程一直运行）
            sup._is_child_alive = lambda: True  # type: ignore[assignment]

            sup._terminate_child()
            # SIGTERM 后超时 -> 应触发 kill()
            popen.kill.assert_called()


if __name__ == "__main__":
    unittest.main()