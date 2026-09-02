"""Tests for memory optimizer."""

import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import pytest

from mind.optimizer import MemoryOptimizer, MemoryOptimizerLoop, OptimizerBusyError


@pytest.fixture
def temp_memory_dir():
    """Create a temporary memory directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def optimizer(temp_memory_dir):
    """Create a MemoryOptimizer instance."""
    return MemoryOptimizer(temp_memory_dir, interval_seconds=3600)


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer class."""
    
    def test_init_default_values(self, temp_memory_dir):
        """Test default initialization."""
        opt = MemoryOptimizer(temp_memory_dir)
        assert opt.interval_seconds == 64800
        assert not opt.is_running
    
    def test_init_custom_interval(self, temp_memory_dir):
        """Test custom interval initialization."""
        opt = MemoryOptimizer(temp_memory_dir, interval_seconds=7200)
        assert opt.interval_seconds == 7200
    
    def test_init_minimum_interval(self, temp_memory_dir):
        """Test minimum interval enforcement."""
        opt = MemoryOptimizer(temp_memory_dir, interval_seconds=100)
        assert opt.interval_seconds == 300  # Min 5 minutes
    
    def test_read_pending_with_facts(self, optimizer, temp_memory_dir):
        """Test reading pending file with facts."""
        pending_file = temp_memory_dir / "PENDING.md"
        pending_file.write_text("- 事实1\n- 事实2\n- 事实3\n")
        
        facts = optimizer._read_pending()
        assert len(facts) == 3
        assert "事实1" in facts
    
    def test_filter_facts_short_facts(self, optimizer):
        """Test filtering out short facts."""
        facts = ["短", "太短了", "这是一个足够长的事实描述"]
        worthy = optimizer._filter_facts(facts)
        assert "这是一个足够长的事实描述" in worthy
        assert "短" not in worthy
    
    def test_filter_facts_duplicates(self, optimizer, temp_memory_dir):
        """Test filtering out duplicate facts."""
        memory_file = temp_memory_dir / "MEMORY.md"
        memory_file.write_text("- 这是一条已有的记忆内容\n")
        facts = ["这是一条已有的记忆内容", "这是一条全新的记忆内容"]
        worthy = optimizer._filter_facts(facts)
        assert "这是一条全新的记忆内容" in worthy
        assert "这是一条已有的记忆内容" not in worthy
    
    def test_filter_facts_temporal_markers(self, optimizer):
        """Test filtering out temporal markers."""
        facts = ["今天天气好", "最近很忙", "长期偏好Python"]
        worthy = optimizer._filter_facts(facts)
        assert "长期偏好Python" in worthy
        assert "今天天气好" not in worthy
    
    def test_archive_facts_success(self, optimizer, temp_memory_dir):
        """Test successful fact archival."""
        facts = ["事实1", "事实2"]
        count = optimizer._archive_facts(facts)
        assert count == 2
        
        memory_file = temp_memory_dir / "MEMORY.md"
        content = memory_file.read_text()
        assert "事实1" in content
        assert "事实2" in content
    
    @pytest.mark.asyncio
    async def test_run_with_pending(self, optimizer, temp_memory_dir):
        """Test run with pending facts."""
        pending_file = temp_memory_dir / "PENDING.md"
        pending_file.write_text("- 这是一条有效的事实记录\n- 这是另一条有效的事实记录\n")
        archived = await optimizer.optimize()
        assert archived == 2
        
        memory_file = temp_memory_dir / "MEMORY.md"
        memory_content = memory_file.read_text()
        assert "这是一条有效的事实记录" in memory_content
        assert "这是另一条有效的事实记录" in memory_content
    
    @pytest.mark.asyncio
    async def test_run_filters_short_facts(self, optimizer, temp_memory_dir):
        """Test run filters out short facts."""
        pending_file = temp_memory_dir / "PENDING.md"
        pending_file.write_text("- 短\n- 这是一条足够长的事实记录\n")
        
        archived = await optimizer.optimize()
        assert archived == 1
    
    @pytest.mark.asyncio
    async def test_run_concurrent_prevention(self, optimizer, temp_memory_dir):
        """Test concurrent run prevention using a lock."""
        # Manually acquire the lock to simulate a running optimization
        async with optimizer._lock:
            # While lock is held, second call should raise OptimizerBusyError
            with pytest.raises(OptimizerBusyError):
                await optimizer.optimize()
class TestMemoryOptimizerLoop:
    """Tests for MemoryOptimizerLoop class."""
    
    def test_init(self, optimizer):
        """Test loop initialization."""
        loop = MemoryOptimizerLoop(optimizer, interval_seconds=3600)
        assert loop._optimizer == optimizer
        assert loop._interval == 3600  # 1 hour = 3600 seconds
        assert not loop._running
    
    def test_init_minimum_interval(self, optimizer):
        """Test minimum interval enforcement."""
        loop = MemoryOptimizerLoop(optimizer, interval_seconds=100)
        assert loop._interval == 300  # Min 5 minutes
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, optimizer):
        """Test start and stop."""
        loop = MemoryOptimizerLoop(optimizer, interval_seconds=1)
        
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        loop._running = False
        await task
        
        assert not loop._running
    
    def test_next_fire_time(self, optimizer):
        """Test next fire time calculation."""
        loop = MemoryOptimizerLoop(optimizer, interval_seconds=3600)
        
        seconds = loop._seconds_until_next_tick()
        
        # Should return a positive number of seconds
        assert seconds > 0
        # Should be less than or equal to the interval
        assert seconds <= 3600
