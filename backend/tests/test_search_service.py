import pytest
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from services.search_service import SearchService, SearchConfig, SearchResult
from services.vector_store_service import VectorStoreService, VectorStoreConfig
from services.embedding_service import EmbeddingService, EmbeddingConfig

@pytest.fixture
async def vector_store():
    """创建向量存储服务"""
    config = VectorStoreConfig(
        persist_directory="test_vector_store",
        allow_reset=True
    )
    service = VectorStoreService(config)
    yield service
    # 清理测试数据
    await service.reset()

@pytest.fixture
async def embedding_service():
    """创建向量化服务"""
    config = EmbeddingConfig()
    service = EmbeddingService(config)
    return service

@pytest.fixture
async def search_service(vector_store, embedding_service):
    """创建搜索服务"""
    config = SearchConfig(
        top_k=3,
        score_threshold=0.5,
        max_results=5,
        rerank=True,
        rerank_top_k=3,
        rerank_threshold=0.6
    )
    service = SearchService(vector_store, embedding_service, config)
    return service

@pytest.fixture
def sample_documents():
    """创建示例文档"""
    return [
        "这是第一个测试文档",
        "这是第二个测试文档",
        "这是第三个测试文档",
        "这是第四个测试文档",
        "这是第五个测试文档"
    ]

@pytest.fixture
def sample_vectors():
    """创建示例向量"""
    return [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7]
    ]

@pytest.fixture
def sample_metadata():
    """创建示例元数据"""
    return [
        {"source": "doc1", "page": 1},
        {"source": "doc2", "page": 2},
        {"source": "doc3", "page": 3},
        {"source": "doc4", "page": 4},
        {"source": "doc5", "page": 5}
    ]

@pytest.mark.asyncio
async def test_search(
    search_service,
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试搜索功能"""
    # 创建测试集合
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加测试数据
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 执行搜索
    query = "测试文档"
    results = await search_service.search(
        query=query,
        collection_name=collection_name
    )
    
    # 验证结果
    assert len(results) > 0
    assert all(isinstance(result, SearchResult) for result in results)
    assert all(result.score >= search_service.config.score_threshold for result in results)

@pytest.mark.asyncio
async def test_search_with_filters(
    search_service,
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试带过滤条件的搜索"""
    # 创建测试集合
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加测试数据
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 执行带过滤条件的搜索
    query = "测试文档"
    filters = {"page": {"$in": [1, 2]}}
    results = await search_service.search(
        query=query,
        collection_name=collection_name,
        filters=filters
    )
    
    # 验证结果
    assert len(results) > 0
    assert all(result.metadata["page"] in [1, 2] for result in results)

@pytest.mark.asyncio
async def test_search_history(
    search_service,
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试搜索历史"""
    # 创建测试集合
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加测试数据
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 执行搜索
    query = "测试文档"
    results = await search_service.search(
        query=query,
        collection_name=collection_name
    )
    
    # 保存搜索历史
    await search_service.save_search_history(
        collection_name=collection_name,
        query=query,
        results=results
    )
    
    # 获取搜索历史
    history = await search_service.get_search_history(collection_name)
    
    # 验证历史记录
    assert len(history) > 0
    assert history[-1]["query"] == query
    assert len(history[-1]["results"]) == len(results)

@pytest.mark.asyncio
async def test_search_stats(
    search_service,
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试搜索统计信息"""
    # 创建测试集合
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加测试数据
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 执行多次搜索
    queries = ["测试文档1", "测试文档2", "测试文档3"]
    for query in queries:
        results = await search_service.search(
            query=query,
            collection_name=collection_name
        )
        await search_service.save_search_history(
            collection_name=collection_name,
            query=query,
            results=results
        )
    
    # 获取统计信息
    stats = await search_service.get_search_stats(collection_name)
    
    # 验证统计信息
    assert stats["total_searches"] == len(queries)
    assert stats["avg_results"] > 0
    assert stats["avg_score"] > 0
    assert stats["last_search"] is not None

@pytest.mark.asyncio
async def test_clear_cache(search_service):
    """测试清理缓存"""
    # 清理特定集合的缓存
    await search_service.clear_cache("test_collection")
    
    # 清理所有缓存
    await search_service.clear_cache() 