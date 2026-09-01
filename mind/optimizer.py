"""Memory optimizer - automatic archival of PENDING facts to MEMORY.

Periodically processes accumulated facts in PENDING.md, validates them,
and archives worthy ones to MEMORY.md with quality filtering.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class OptimizerBusyError(RuntimeError):
    """Raised when optimizer is already running."""
    pass


class MemoryOptimizer:
    """Automatic memory archival from PENDING to MEMORY.
    
    Periodically processes accumulated facts, validates them against
    quality criteria, and archives worthy ones to long-term memory.
    """
    
    def __init__(
        self,
        memory_dir: Path,
        interval_seconds: int = 64800,  # 18 hours default
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._memory_dir = memory_dir
        self._pending_file = memory_dir / "PENDING.md"
        self._memory_file = memory_dir / "MEMORY.md"
        self._interval = max(300, interval_seconds)  # Min 5 minutes
        self._now_fn = _now_fn or datetime.now
        self._lock = asyncio.Lock()
        self._running = False
        
    @property
    def is_running(self) -> bool:
        """Check if optimizer is currently running."""
        return self._lock.locked()
    
    @property
    def interval_seconds(self) -> int:
        """Get the archival interval in seconds."""
        return self._interval
    
    async def optimize(self) -> int:
        """Run one optimization cycle.
        
        Returns:
            Number of facts archived (0 if skipped or failed).
        
        Raises:
            OptimizerBusyError: If another optimization is in progress.
        """
        if self._lock.locked():
            raise OptimizerBusyError("Memory optimizer is already running")
        
        async with self._lock:
            return await self._do_optimize()
    
    async def _do_optimize(self) -> int:
        """Internal optimization logic."""
        try:
            # 1. Read pending facts
            pending_facts = self._read_pending()
            if not pending_facts:
                logger.debug("[optimizer] No pending facts, skipping")
                return 0
            
            # 2. Validate and filter facts
            worthy_facts = self._filter_facts(pending_facts)
            if not worthy_facts:
                logger.info(
                    "[optimizer] No worthy facts after filtering (%d pending)",
                    len(pending_facts),
                )
                # Clear pending file since all facts were filtered
                self._clear_pending()
                return 0
            
            # 3. Archive to MEMORY
            archived_count = self._archive_facts(worthy_facts)
            
            # 4. Clear pending file
            self._clear_pending()
            
            # 5. Emit event (if events module available)
            try:
                from mind.extensions.events import emit_consolidation_committed
                await emit_consolidation_committed(
                    archived_count=archived_count,
                    total_pending=len(pending_facts),
                )
            except ImportError:
                pass  # Events module not yet implemented
            
            logger.info(
                "[optimizer] Archived %d/%d facts to MEMORY",
                archived_count,
                len(pending_facts),
            )
            
            return archived_count
            
        except Exception as e:
            logger.exception("[optimizer] Optimization failed: %s", e)
            return 0
    
    def _read_pending(self) -> list[str]:
        """Read facts from PENDING.md."""
        if not self._pending_file.exists():
            return []
        
        facts = []
        with open(self._pending_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    facts.append(line)
        
        return facts
    
    def _filter_facts(self, facts: list[str]) -> list[str]:
        """Filter facts based on quality criteria.
        
        Rules:
        - Must be at least 10 characters
        - Must not be empty or whitespace only
        - Must not be duplicate of existing memory
        - Should not contain temporal markers (今天, 昨天, 最近, etc.)
        """
        existing = self._read_memory()
        existing_set = {f.strip().lower() for f in existing}
        
        worthy = []
        temporal_markers = [
            "今天", "昨天", "前天", "最近", "这几天", "本周", "上周",
            "today", "yesterday", "recently", "this week", "last week",
        ]
        
        for fact in facts:
            fact_stripped = fact.strip()
            
            # Length check
            if len(fact_stripped) < 10:
                continue
            
            # Duplicate check
            if fact_stripped.lower() in existing_set:
                continue
            
            # Temporal marker check
            fact_lower = fact_stripped.lower()
            if any(marker in fact_lower for marker in temporal_markers):
                continue
            
            worthy.append(fact_stripped)
            existing_set.add(fact_stripped.lower())
        
        return worthy
    
    def _read_memory(self) -> list[str]:
        """Read existing facts from MEMORY.md."""
        if not self._memory_file.exists():
            return []
        
        facts = []
        with open(self._memory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    facts.append(line)
        
        return facts
    
    def _archive_facts(self, facts: list[str]) -> int:
        """Append facts to MEMORY.md.
        
        Returns:
            Number of facts archived.
        """
        if not facts:
            return 0
        
        # Ensure MEMORY.md exists
        if not self._memory_file.exists():
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._memory_file, "w", encoding="utf-8") as f:
                f.write("# 长期记忆\n\n")
        
        # Append new facts
        with open(self._memory_file, "a", encoding="utf-8") as f:
            f.write("\n")
            for fact in facts:
                f.write(f"{fact}\n")
        
        return len(facts)
    
    def _clear_pending(self) -> None:
        """Clear PENDING.md file."""
        if self._pending_file.exists():
            with open(self._pending_file, "w", encoding="utf-8") as f:
                f.write("# 待归档事实\n\n由对话自动提取，等待优化器处理。\n\n")


class MemoryOptimizerLoop:
    """Background loop that periodically runs memory optimization."""
    
    def __init__(
        self,
        optimizer: MemoryOptimizer | None,
        interval_seconds: int = 64800,  # 18 hours default
        _now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._optimizer = optimizer
        self._interval = max(300, interval_seconds)  # Min 5 minutes
        self._now_fn = _now_fn or datetime.now
        self._running = False
    
    async def run(self) -> None:
        """Start the optimization loop."""
        self._running = True
        logger.info(
            "[optimizer] Starting optimization loop (interval=%ds / %.1fh)",
            self._interval,
            self._interval / 3600,
        )
        
        while self._running:
            # Calculate seconds until next tick (aligned to interval)
            secs = self._seconds_until_next_tick()
            logger.debug("[optimizer] Next optimization in %.0f seconds", secs)
            
     
