"""Supervisor 进程管理 —— 父进程托管 gateway 子进程，支持安全自重启。

工作原理：
  - app.py 无参数启动时，检测环境变量 PROACTIVEMIND_CHILD：
      * 未设置 → 父进程模式：Supervisor 启动 gateway 子进程并按需重启
      * 已设置 → 子进程模式：直接运行对应 gateway（web / telegram）
  - PID 文件锁防止同一 workspace 启动多个 Supervisor
  - 子进程异常退出时按指数退避策略重启，达到上限后 Supervisor 退出
  - SIGTERM 优雅关闭（先 SIGTERM，超时再 SIGKILL）

启动方式（由 app.py 调度）：
  python app.py                  -> 父进程（supervisor）模式
  python app.py web              -> 直接 web（跳过 supervisor）
  python app.py telegram         -> 直接 telegram（跳过 supervisor）
  python app.py supervise web    -> 父进程模式托管 web gateway
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

CHILD_ENV_VAR = "PROACTIVEMIND_CHILD"
SUPERVISOR_PID_FILE = "supervisor.pid"
SUPERVISOR_LOCK_FILE = "supervisor.lock"

DEFAULT_MAX_RESTARTS = 5
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_MAX = 30.0
DEFAULT_SHUTDOWN_TIMEOUT = 10.0


class SupervisorError(RuntimeError):
    """Supervisor 流程中的错误。"""

    pass


@dataclass
class SupervisorConfig:
    """Supervisor 行为配置。"""

    workspace: Path
    gateway_mode: str  # "web" | "telegram"
    max_restarts: int = DEFAULT_MAX_RESTARTS
    backoff_base: float = DEFAULT_BACKOFF_BASE
    backoff_max: float = DEFAULT_BACKOFF_MAX
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT
    python_executable: str | None = None
    app_entrypoint: str | None = None

    def __post_init__(self) -> None:
        if self.gateway_mode not in ("web", "telegram"):
            raise ValueError(
                f"gateway_mode 必须是 web 或 telegram，得到 {self.gateway_mode}"
            )
        if self.max_restarts < 0:
            raise ValueError("max_restarts 必须 >= 0")
        if self.backoff_base <= 0:
            raise ValueError("backoff_base 必须 > 0")


@dataclass
class ChildState:
    """子进程运行时状态。"""

    pid: int | None = None
    started_at: float | None = None
    exit_code: int | None = None
    restart_count: int = 0
    next_backoff: float = 0.0


class PidFileLock:
    """简单的 PID 文件锁（基于文件存在性 + 内容校验）。

    不是严格的 OS 锁，但足够防止同一 workspace 启动两个 Supervisor。
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def acquire(self) -> bool:
        """尝试获取锁。返回 True 表示成功，False 表示已存在。"""
        if self.path.exists():
            existing_pid = self._read_pid()
            if existing_pid is not None and self._pid_alive(existing_pid):
                return False
            # 锁文件过期（残留），删除后重新创建
            try:
                self.path.unlink()
            except OSError:
                return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        return True

    def release(self) -> None:
        """释放锁（仅当锁属于自己时）。"""
        if not self.path.exists():
            return
        existing = self._read_pid()
        if existing == os.getpid():
            try:
                self.path.unlink()
            except OSError:
                pass

    def _read_pid(self) -> int | None:
        try:
            text = self.path.read_text(encoding="utf-8").strip()
            return int(text) if text else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if sys.platform == "win32":
            # Windows 下没有 os.kill(pid, 0)，使用 OpenProcess
            try:
                import ctypes

                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                STILL_ACTIVE = 259
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, pid
                )
                if not handle:
                    return False
                try:
                    exit_code = ctypes.c_ulong()
                    ok = ctypes.windll.kernel32.GetExitCodeProcess(
                        handle, ctypes.byref(exit_code)
                    )
                    return bool(ok) and exit_code.value == STILL_ACTIVE
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            except OSError:
                return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


class Supervisor:
    """父进程管理器：启动并守护 gateway 子进程。"""

    def __init__(
        self,
        config: SupervisorConfig,
        *,
        spawn_fn: Callable[[SupervisorConfig], subprocess.Popen] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._spawn_fn = spawn_fn or self._default_spawn
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn
        self._child = ChildState()
        self._running = False
        self._stop_requested = False
        self._lock = PidFileLock(config.workspace / SUPERVISOR_LOCK_FILE)

    def acquire_lock(self) -> None:
        """获取 PID 锁。失败时抛出 SupervisorError。"""
        if not self._lock.acquire():
            raise SupervisorError(
                f"Supervisor 已在运行（锁文件: {self._lock.path}）"
            )

    def release_lock(self) -> None:
        self._lock.release()

    def _default_spawn(self, config: SupervisorConfig) -> subprocess.Popen:
        """启动 gateway 子进程。"""
        executable = config.python_executable or sys.executable
        entry = config.app_entrypoint or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "app.py")
        )
        env = os.environ.copy()
        env[CHILD_ENV_VAR] = "1"
        # Windows 下需要 CREATE_NEW_PROCESS_GROUP 以便发 SIGTERM
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            )
        log.info(
            "Supervisor 启动子进程 mode=%s pid_将获取",
            config.gateway_mode,
        )
        return subprocess.Popen(
            [executable, entry, config.gateway_mode],
            env=env,
            **kwargs,
        )

    def request_stop(self) -> None:
        """请求优雅停止。"""
        self._stop_requested = True

    def _wait_child_exit(self, timeout: float) -> bool:
        """等待子进程退出。返回是否在超时内退出。"""
        if self._child.pid is None:
            return True
        deadline = self._now_fn() + timeout
        while self._now_fn() < deadline:
            if self._is_child_alive():
                self._sleep_fn(0.1)
            else:
                return True
        return not self._is_child_alive()

    def _is_child_alive(self) -> bool:
        if self._child.pid is None:
            return False
        try:
            # 在 Windows 上使用 poll() 检查子进程状态
            if sys.platform == "win32":
                # 我们用 popen 句柄无法直接复用，存为子属性
                handle = getattr(self, "_popen_handle", None)
                if handle is not None and handle.poll() is None:
                    return True
                return False
            os.kill(self._child.pid, 0)
            return True
        except OSError:
            return False

    def _terminate_child(self) -> None:
        """优雅终止子进程：先 SIGTERM，超时再 SIGKILL。"""
        if self._child.pid is None:
            return
        log.info("Supervisor 终止子进程 pid=%s", self._child.pid)
        try:
            if sys.platform == "win32":
                self._child.pid and subprocess.Popen  # noqa: B015
                # 通过 CTRL_BREAK_EVENT（要求 CREATE_NEW_PROCESS_GROUP）
                handle = getattr(self, "_popen_handle", None)
                if handle is not None:
                    handle.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.kill(self._child.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, AttributeError) as exc:
            log.warning("发送终止信号失败: %s", exc)

        if self._wait_child_exit(self.config.shutdown_timeout):
            return

        log.warning("子进程未在 %.1fs 内退出，强制 SIGKILL", self.config.shutdown_timeout)
        try:
            handle = getattr(self, "_popen_handle", None)
            if handle is not None:
                handle.kill()
            elif sys.platform != "win32" and self._child.pid:
                os.kill(self._child.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError, AttributeError) as exc:
            log.warning("强制终止失败: %s", exc)

    def _compute_backoff(self, restart_count: int) -> float:
        """指数退避：base * 2^restart_count，上限 backoff_max。"""
        delay = self.config.backoff_base * (2 ** restart_count)
        return min(delay, self.config.backoff_max)

    def _spawn(self) -> subprocess.Popen:
        """启动子进程并记录状态。"""
        popen = self._spawn_fn(self.config)
        self._popen_handle = popen
        self._child.pid = popen.pid
        self._child.started_at = self._now_fn()
        self._child.exit_code = None
        self._child.next_backoff = 0.0
        log.info("子进程已启动 pid=%s mode=%s", popen.pid, self.config.gateway_mode)
        return popen

    def _reap(self, popen: subprocess.Popen) -> int:
        """回收子进程并返回退出码。"""
        try:
            exit_code = popen.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            log.warning("wait() 超时，使用 poll()")
            exit_code = popen.poll() or -1
        self._child.exit_code = exit_code
        log.info("子进程退出 pid=%s exit_code=%s", self._child.pid, exit_code)
        return exit_code

    def run(self) -> int:
        """主循环：启动、监控、重启。返回最终退出码。"""
        self._running = True
        log.info(
            "Supervisor 启动 mode=%s max_restarts=%s",
            self.config.gateway_mode, self.config.max_restarts,
        )

        while self._running and not self._stop_requested:
            popen = self._spawn()
            exit_code = self._reap(popen)

            if self._stop_requested:
                log.info("Supervisor 收到停止请求，退出")
                break

            if exit_code == 0:
                # 干净退出 → 正常结束，Supervisor 跟随退出
                log.info("子进程干净退出，Supervisor 退出")
                self._running = False
                return 0

            self._child.restart_count += 1
            if self._child.restart_count > self.config.max_restarts:
                log.error(
                    "子进程连续重启 %d 次达到上限，Supervisor 退出",
                    self._child.restart_count,
                )
                return exit_code

            delay = self._compute_backoff(self._child.restart_count - 1)
            self._child.next_backoff = delay
            log.warning(
                "子进程异常退出，%.1fs 后重启（第 %d/%d 次）",
                delay, self._child.restart_count, self.config.max_restarts,
            )
            self._sleep_fn(delay)

        self._running = False
        return 0

    def stop(self) -> None:
        """请求停止并立即终止子进程（同步）。"""
        self._stop_requested = True
        self._running = False
        self._terminate_child()


def is_child_process() -> bool:
    """判断当前进程是否是被 Supervisor 启动的子进程。"""
    return os.environ.get(CHILD_ENV_VAR) == "1"


def supervise(gateway_mode: str, workspace: Path) -> int:
    """便捷入口：直接以 Supervisor 模式启动。"""
    config = SupervisorConfig(workspace=workspace, gateway_mode=gateway_mode)
    supervisor = Supervisor(config)
    supervisor.acquire_lock()
    try:
        # 安装信号处理器：收到 SIGTERM/SIGINT 时优雅停止
        def _on_signal(signum: int, frame: object) -> None:
            log.info("Supervisor 收到信号 %s", signum)
            supervisor.request_stop()
            supervisor._terminate_child()

        try:
            signal.signal(signal.SIGTERM, _on_signal)
        except (ValueError, OSError):
            # 子线程或非主线程无法注册信号
            pass
        try:
            signal.signal(signal.SIGINT, _on_signal)
        except (ValueError, OSError):
            pass

        return supervisor.run()
    finally:
        supervisor.release_lock()