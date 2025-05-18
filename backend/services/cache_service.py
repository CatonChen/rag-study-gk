from typing import Any, Dict, Optional, Union
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import aiofiles
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheConfig(BaseModel):
    """缓存配置"""
    cache_dir: str = "cache"
    default_ttl: int = 3600  # 默认缓存时间（秒）
    max_size: int = 1000  # 最大缓存条目数
    cleanup_interval: int = 300  # 清理间隔（秒）

class CacheService:
    """缓存服务"""
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None
    ):
        self.config = config or CacheConfig()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._cleanup_task = None
    
    async def start(self):
        """启动缓存服务"""
        # 加载持久化缓存
        await self._load_cache()
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """停止缓存服务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 保存缓存
        await self._save_cache()
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Any: 缓存值
        """
        try:
            # 检查内存缓存
            if key in self._cache:
                cache_item = self._cache[key]
                if not self._is_expired(cache_item):
                    return cache_item["value"]
                else:
                    del self._cache[key]
            
            # 检查持久化缓存
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                    cache_item = json.loads(await f.read())
                
                if not self._is_expired(cache_item):
                    # 更新内存缓存
                    self._cache[key] = cache_item
                    return cache_item["value"]
                else:
                    # 删除过期缓存
                    cache_file.unlink()
            
            return default
            
        except Exception as e:
            logger.error(f"获取缓存失败: {str(e)}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 缓存时间（秒）
        """
        try:
            # 检查缓存大小
            if len(self._cache) >= self.config.max_size:
                await self._cleanup()
            
            # 创建缓存项
            cache_item = {
                "value": value,
                "created_at": datetime.now().isoformat(),
                "expires_at": (
                    datetime.now() + timedelta(seconds=ttl or self.config.default_ttl)
                ).isoformat()
            }
            
            # 更新内存缓存
            self._cache[key] = cache_item
            
            # 更新持久化缓存
            cache_file = self._get_cache_file(key)
            async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache_item, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"设置缓存失败: {str(e)}")
    
    async def delete(self, key: str):
        """
        删除缓存值
        
        Args:
            key: 缓存键
        """
        try:
            # 删除内存缓存
            if key in self._cache:
                del self._cache[key]
            
            # 删除持久化缓存
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
            
        except Exception as e:
            logger.error(f"删除缓存失败: {str(e)}")
    
    async def clear(self):
        """清空缓存"""
        try:
            # 清空内存缓存
            self._cache.clear()
            
            # 清空持久化缓存
            for cache_file in self._cache_dir.glob("*.json"):
                cache_file.unlink()
            
        except Exception as e:
            logger.error(f"清空缓存失败: {str(e)}")
    
    async def get_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            # 清理过期缓存
            await self._cleanup()
            
            return {
                "size": len(self._cache),
                "max_size": self.config.max_size,
                "default_ttl": self.config.default_ttl,
                "cleanup_interval": self.config.cleanup_interval
            }
            
        except Exception as e:
            logger.error(f"获取缓存统计信息失败: {str(e)}")
            return {
                "size": 0,
                "max_size": self.config.max_size,
                "default_ttl": self.config.default_ttl,
                "cleanup_interval": self.config.cleanup_interval
            }
    
    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用MD5哈希作为文件名
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{hash_key}.json"
    
    def _is_expired(self, cache_item: Dict) -> bool:
        """检查缓存是否过期"""
        expires_at = datetime.fromisoformat(cache_item["expires_at"])
        return datetime.now() > expires_at
    
    async def _cleanup(self):
        """清理过期缓存"""
        try:
            # 清理内存缓存
            expired_keys = [
                key for key, item in self._cache.items()
                if self._is_expired(item)
            ]
            for key in expired_keys:
                del self._cache[key]
            
            # 清理持久化缓存
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                        cache_item = json.loads(await f.read())
                    
                    if self._is_expired(cache_item):
                        cache_file.unlink()
                except Exception:
                    cache_file.unlink()
            
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}")
    
    async def _cleanup_loop(self):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理任务失败: {str(e)}")
    
    async def _load_cache(self):
        """加载持久化缓存"""
        try:
            for cache_file in self._cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                        cache_item = json.loads(await f.read())
                    
                    if not self._is_expired(cache_item):
                        # 从文件名恢复键
                        key = cache_file.stem
                        self._cache[key] = cache_item
                    else:
                        cache_file.unlink()
                except Exception:
                    cache_file.unlink()
            
        except Exception as e:
            logger.error(f"加载缓存失败: {str(e)}")
    
    async def _save_cache(self):
        """保存缓存到磁盘"""
        try:
            for key, cache_item in self._cache.items():
                if not self._is_expired(cache_item):
                    cache_file = self._get_cache_file(key)
                    async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(cache_item, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}") 