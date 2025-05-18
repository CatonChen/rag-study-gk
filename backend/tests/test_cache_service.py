import pytest
from pathlib import Path
import json
import asyncio
from datetime import datetime, timedelta
from services.cache_service import CacheService, CacheConfig

@pytest.fixture
async def cache_service():
    """创建缓存服务"""
    config = CacheConfig(
        cache_dir="test_cache",
        default_ttl=1,  # 1秒过期
        max_size=10,
        cleanup_interval=1
    )
    service = CacheService(config)
    await service.start()
    yield service
    await service.stop()
    # 清理测试数据
    await service.clear()

@pytest.mark.asyncio
async def test_set_get(cache_service):
    """测试设置和获取缓存"""
    # 设置缓存
    key = "test_key"
    value = {"test": "value"}
    await cache_service.set(key, value)
    
    # 获取缓存
    result = await cache_service.get(key)
    
    # 验证结果
    assert result == value

@pytest.mark.asyncio
async def test_get_default(cache_service):
    """测试获取默认值"""
    # 获取不存在的缓存
    key = "non_existent_key"
    default = "default_value"
    result = await cache_service.get(key, default)
    
    # 验证结果
    assert result == default

@pytest.mark.asyncio
async def test_delete(cache_service):
    """测试删除缓存"""
    # 设置缓存
    key = "test_key"
    value = {"test": "value"}
    await cache_service.set(key, value)
    
    # 删除缓存
    await cache_service.delete(key)
    
    # 验证结果
    result = await cache_service.get(key)
    assert result is None

@pytest.mark.asyncio
async def test_clear(cache_service):
    """测试清空缓存"""
    # 设置多个缓存
    for i in range(5):
        await cache_service.set(f"key_{i}", f"value_{i}")
    
    # 清空缓存
    await cache_service.clear()
    
    # 验证结果
    for i in range(5):
        result = await cache_service.get(f"key_{i}")
        assert result is None

@pytest.mark.asyncio
async def test_expiration(cache_service):
    """测试缓存过期"""
    # 设置短期缓存
    key = "test_key"
    value = {"test": "value"}
    await cache_service.set(key, value, ttl=1)  # 1秒过期
    
    # 等待过期
    await asyncio.sleep(1.1)
    
    # 验证结果
    result = await cache_service.get(key)
    assert result is None

@pytest.mark.asyncio
async def test_max_size(cache_service):
    """测试最大缓存大小"""
    # 设置超过最大大小的缓存
    for i in range(cache_service.config.max_size + 5):
        await cache_service.set(f"key_{i}", f"value_{i}")
    
    # 验证结果
    stats = await cache_service.get_stats()
    assert stats["size"] <= cache_service.config.max_size

@pytest.mark.asyncio
async def test_get_stats(cache_service):
    """测试获取统计信息"""
    # 设置一些缓存
    for i in range(5):
        await cache_service.set(f"key_{i}", f"value_{i}")
    
    # 获取统计信息
    stats = await cache_service.get_stats()
    
    # 验证结果
    assert stats["size"] == 5
    assert stats["max_size"] == cache_service.config.max_size
    assert stats["default_ttl"] == cache_service.config.default_ttl
    assert stats["cleanup_interval"] == cache_service.config.cleanup_interval

@pytest.mark.asyncio
async def test_persistence(cache_service):
    """测试缓存持久化"""
    # 设置缓存
    key = "test_key"
    value = {"test": "value"}
    await cache_service.set(key, value)
    
    # 停止服务
    await cache_service.stop()
    
    # 重新启动服务
    await cache_service.start()
    
    # 验证结果
    result = await cache_service.get(key)
    assert result == value

@pytest.mark.asyncio
async def test_cleanup_task(cache_service):
    """测试清理任务"""
    # 设置短期缓存
    for i in range(5):
        await cache_service.set(f"key_{i}", f"value_{i}", ttl=1)
    
    # 等待过期
    await asyncio.sleep(1.1)
    
    # 验证结果
    stats = await cache_service.get_stats()
    assert stats["size"] == 0

@pytest.mark.asyncio
async def test_concurrent_access(cache_service):
    """测试并发访问"""
    # 并发设置缓存
    async def set_cache(i):
        await cache_service.set(f"key_{i}", f"value_{i}")
    
    tasks = [set_cache(i) for i in range(10)]
    await asyncio.gather(*tasks)
    
    # 验证结果
    stats = await cache_service.get_stats()
    assert stats["size"] == 10
    
    # 并发获取缓存
    async def get_cache(i):
        return await cache_service.get(f"key_{i}")
    
    results = await asyncio.gather(*[get_cache(i) for i in range(10)])
    assert all(result is not None for result in results) 