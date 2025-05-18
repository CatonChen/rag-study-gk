from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreConfig(BaseModel):
    """向量存储配置"""
    persist_directory: str = "03-vector-store"
    collection_metadata: Optional[Dict] = None
    embedding_function: Optional[str] = "default"
    distance_metric: str = "cosine"
    allow_reset: bool = False

class VectorStoreService:
    """向量存储服务"""
    
    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None
    ):
        self.config = config or VectorStoreConfig()
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=self.config.allow_reset
            )
        )
        self._collections: Dict[str, Any] = {}
    
    async def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        创建向量集合
        
        Args:
            collection_name: 集合名称
            metadata: 集合元数据
            
        Returns:
            str: 集合名称
        """
        try:
            # 合并配置元数据和自定义元数据
            collection_metadata = {
                **(self.config.collection_metadata or {}),
                **(metadata or {}),
                "created_at": datetime.now().isoformat()
            }
            
            # 创建集合
            collection = self.client.create_collection(
                name=collection_name,
                metadata=collection_metadata,
                embedding_function=self._get_embedding_function()
            )
            
            self._collections[collection_name] = collection
            return collection_name
            
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            raise
    
    async def get_collection(
        self,
        collection_name: str
    ) -> Any:
        """
        获取向量集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Any: 集合对象
        """
        try:
            if collection_name not in self._collections:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self._get_embedding_function()
                )
                self._collections[collection_name] = collection
            
            return self._collections[collection_name]
            
        except Exception as e:
            logger.error(f"获取集合失败: {str(e)}")
            raise
    
    async def delete_collection(
        self,
        collection_name: str
    ):
        """
        删除向量集合
        
        Args:
            collection_name: 集合名称
        """
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
                
        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            raise
    
    async def list_collections(self) -> List[Dict]:
        """
        列出所有向量集合
        
        Returns:
            List[Dict]: 集合信息列表
        """
        try:
            collections = self.client.list_collections()
            return [
                {
                    "name": collection.name,
                    "metadata": collection.metadata,
                    "count": collection.count()
                }
                for collection in collections
            ]
            
        except Exception as e:
            logger.error(f"列出集合失败: {str(e)}")
            raise
    
    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        添加向量
        
        Args:
            collection_name: 集合名称
            vectors: 向量列表
            documents: 文档列表
            metadatas: 元数据列表
            ids: ID列表
        """
        try:
            collection = await self.get_collection(collection_name)
            
            # 生成ID
            if ids is None:
                ids = [f"{collection_name}_{i}" for i in range(len(vectors))]
            
            # 添加向量
            collection.add(
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            logger.error(f"添加向量失败: {str(e)}")
            raise
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        搜索向量
        
        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        try:
            collection = await self.get_collection(collection_name)
            
            # 执行搜索
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filters
            )
            
            # 格式化结果
            search_results = []
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i],
                    "source": collection_name
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"搜索向量失败: {str(e)}")
            raise
    
    async def update_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        更新向量
        
        Args:
            collection_name: 集合名称
            vectors: 向量列表
            documents: 文档列表
            metadatas: 元数据列表
            ids: ID列表
        """
        try:
            collection = await self.get_collection(collection_name)
            
            # 更新向量
            collection.update(
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            logger.error(f"更新向量失败: {str(e)}")
            raise
    
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ):
        """
        删除向量
        
        Args:
            collection_name: 集合名称
            ids: ID列表
        """
        try:
            collection = await self.get_collection(collection_name)
            
            # 删除向量
            collection.delete(ids=ids)
            
        except Exception as e:
            logger.error(f"删除向量失败: {str(e)}")
            raise
    
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict: 统计信息
        """
        try:
            collection = await self.get_collection(collection_name)
            
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {str(e)}")
            raise
    
    def _get_embedding_function(self):
        """获取嵌入函数"""
        if self.config.embedding_function == "default":
            return embedding_functions.DefaultEmbeddingFunction()
        else:
            raise ValueError(f"不支持的嵌入函数: {self.config.embedding_function}")
    
    async def reset(self):
        """重置向量存储"""
        if not self.config.allow_reset:
            raise ValueError("重置功能未启用")
        
        try:
            self.client.reset()
            self._collections.clear()
            
        except Exception as e:
            logger.error(f"重置向量存储失败: {str(e)}")
            raise 