from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import json
import numpy as np
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """搜索结果模型"""
    content: str
    metadata: Dict
    score: float
    source: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None

class SearchConfig(BaseModel):
    """搜索配置"""
    top_k: int = 5  # 返回结果数量
    score_threshold: float = 0.7  # 相似度阈值
    max_results: int = 20  # 最大结果数
    rerank: bool = True  # 是否重排序
    rerank_top_k: int = 10  # 重排序候选数
    rerank_threshold: float = 0.8  # 重排序阈值

class SearchService:
    """检索服务"""
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        config: Optional[SearchConfig] = None
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or SearchConfig()
        self._search_cache: Dict[str, List[SearchResult]] = {}
        self._cache_ttl = 3600  # 缓存有效期（秒）
    
    async def search(
        self,
        query: str,
        collection_name: str,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        执行语义搜索
        
        Args:
            query: 查询文本
            collection_name: 集合名称
            filters: 过滤条件
            **kwargs: 其他参数
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 检查缓存
            cache_key = f"{collection_name}:{query}:{str(filters)}"
            if cache_key in self._search_cache:
                return self._search_cache[cache_key]
            
            # 生成查询向量
            query_vector = await self.embedding_service.get_embedding(query)
            
            # 执行向量搜索
            results = await self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=self.config.max_results,
                filters=filters
            )
            
            # 过滤和重排序
            filtered_results = self._filter_results(results)
            if self.config.rerank:
                reranked_results = await self._rerank_results(query, filtered_results)
            else:
                reranked_results = filtered_results
            
            # 转换为搜索结果
            search_results = [
                SearchResult(
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    source=result["source"],
                    page_number=result.get("page_number"),
                    section_title=result.get("section_title"),
                    chunk_index=result.get("chunk_index")
                )
                for result in reranked_results[:self.config.top_k]
            ]
            
            # 更新缓存
            self._search_cache[cache_key] = search_results
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise
    
    def _filter_results(self, results: List[Dict]) -> List[Dict]:
        """过滤搜索结果"""
        return [
            result for result in results
            if result["score"] >= self.config.score_threshold
        ]
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Dict]
    ) -> List[Dict]:
        """重排序搜索结果"""
        if not results:
            return results
        
        # 获取重排序候选
        candidates = results[:self.config.rerank_top_k]
        
        # 计算重排序分数
        reranked_scores = await self._calculate_rerank_scores(query, candidates)
        
        # 合并原始分数和重排序分数
        for result, rerank_score in zip(candidates, reranked_scores):
            result["score"] = (result["score"] + rerank_score) / 2
        
        # 按新分数排序
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 过滤低分结果
        return [
            result for result in candidates
            if result["score"] >= self.config.rerank_threshold
        ]
    
    async def _calculate_rerank_scores(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[float]:
        """计算重排序分数"""
        # 获取查询和候选文本的嵌入
        query_embedding = await self.embedding_service.get_embedding(query)
        candidate_embeddings = await asyncio.gather(*[
            self.embedding_service.get_embedding(candidate["content"])
            for candidate in candidates
        ])
        
        # 计算余弦相似度
        scores = []
        for candidate_embedding in candidate_embeddings:
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            scores.append(float(similarity))
        
        return scores
    
    async def clear_cache(self, collection_name: Optional[str] = None):
        """清理缓存"""
        if collection_name:
            # 清理特定集合的缓存
            keys_to_remove = [
                key for key in self._search_cache.keys()
                if key.startswith(f"{collection_name}:")
            ]
            for key in keys_to_remove:
                del self._search_cache[key]
        else:
            # 清理所有缓存
            self._search_cache.clear()
    
    async def get_search_history(
        self,
        collection_name: str,
        limit: int = 100
    ) -> List[Dict]:
        """获取搜索历史"""
        try:
            history_file = Path(f"04-search-results/{collection_name}_history.json")
            if not history_file.exists():
                return []
            
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            
            return history[-limit:]
            
        except Exception as e:
            logger.error(f"获取搜索历史失败: {str(e)}")
            return []
    
    async def save_search_history(
        self,
        collection_name: str,
        query: str,
        results: List[SearchResult]
    ):
        """保存搜索历史"""
        try:
            history_file = Path(f"04-search-results/{collection_name}_history.json")
            history_file.parent.mkdir(exist_ok=True)
            
            # 读取现有历史
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
            
            # 添加新记录
            history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "results": [
                    {
                        "content": result.content,
                        "score": result.score,
                        "source": result.source
                    }
                    for result in results
                ]
            })
            
            # 保存历史
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"保存搜索历史失败: {str(e)}")
    
    async def get_search_stats(
        self,
        collection_name: str
    ) -> Dict:
        """获取搜索统计信息"""
        try:
            history = await self.get_search_history(collection_name)
            
            # 计算统计信息
            total_searches = len(history)
            avg_results = sum(len(h["results"]) for h in history) / total_searches if total_searches > 0 else 0
            avg_score = sum(
                sum(r["score"] for r in h["results"]) / len(h["results"])
                for h in history if h["results"]
            ) / total_searches if total_searches > 0 else 0
            
            return {
                "total_searches": total_searches,
                "avg_results": avg_results,
                "avg_score": avg_score,
                "last_search": history[-1]["timestamp"] if history else None
            }
            
        except Exception as e:
            logger.error(f"获取搜索统计信息失败: {str(e)}")
            return {
                "total_searches": 0,
                "avg_results": 0,
                "avg_score": 0,
                "last_search": None
            } 