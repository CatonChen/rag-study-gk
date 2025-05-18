import pytest
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from services.vector_store_service import VectorStoreService, VectorStoreConfig

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
async def test_create_collection(vector_store):
    """测试创建集合"""
    collection_name = "test_collection"
    metadata = {"description": "测试集合"}
    
    # 创建集合
    result = await vector_store.create_collection(
        collection_name=collection_name,
        metadata=metadata
    )
    
    # 验证结果
    assert result == collection_name
    
    # 验证集合存在
    collections = await vector_store.list_collections()
    assert any(c["name"] == collection_name for c in collections)

@pytest.mark.asyncio
async def test_get_collection(vector_store):
    """测试获取集合"""
    collection_name = "test_collection"
    
    # 创建集合
    await vector_store.create_collection(collection_name)
    
    # 获取集合
    collection = await vector_store.get_collection(collection_name)
    
    # 验证结果
    assert collection is not None
    assert collection.name == collection_name

@pytest.mark.asyncio
async def test_delete_collection(vector_store):
    """测试删除集合"""
    collection_name = "test_collection"
    
    # 创建集合
    await vector_store.create_collection(collection_name)
    
    # 删除集合
    await vector_store.delete_collection(collection_name)
    
    # 验证集合不存在
    collections = await vector_store.list_collections()
    assert not any(c["name"] == collection_name for c in collections)

@pytest.mark.asyncio
async def test_list_collections(vector_store):
    """测试列出集合"""
    # 创建多个集合
    collection_names = ["test_collection1", "test_collection2", "test_collection3"]
    for name in collection_names:
        await vector_store.create_collection(name)
    
    # 获取集合列表
    collections = await vector_store.list_collections()
    
    # 验证结果
    assert len(collections) >= len(collection_names)
    assert all(any(c["name"] == name for c in collections) for name in collection_names)

@pytest.mark.asyncio
async def test_add_vectors(
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试添加向量"""
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加向量
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 验证结果
    collection = await vector_store.get_collection(collection_name)
    assert collection.count() == len(sample_vectors)

@pytest.mark.asyncio
async def test_search_vectors(
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试搜索向量"""
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加向量
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 执行搜索
    query_vector = [0.2, 0.3, 0.4]
    results = await vector_store.search(
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=3
    )
    
    # 验证结果
    assert len(results) > 0
    assert all("content" in result for result in results)
    assert all("metadata" in result for result in results)
    assert all("score" in result for result in results)

@pytest.mark.asyncio
async def test_update_vectors(
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试更新向量"""
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加向量
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 更新向量
    updated_vectors = [[v * 2 for v in vec] for vec in sample_vectors[:2]]
    updated_documents = [doc + " (已更新)" for doc in sample_documents[:2]]
    updated_metadata = [{"source": meta["source"], "page": meta["page"], "updated": True} for meta in sample_metadata[:2]]
    ids = [f"{collection_name}_{i}" for i in range(2)]
    
    await vector_store.update_vectors(
        collection_name=collection_name,
        vectors=updated_vectors,
        documents=updated_documents,
        metadatas=updated_metadata,
        ids=ids
    )
    
    # 验证结果
    collection = await vector_store.get_collection(collection_name)
    assert collection.count() == len(sample_vectors)

@pytest.mark.asyncio
async def test_delete_vectors(
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试删除向量"""
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 添加向量
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 删除向量
    ids = [f"{collection_name}_{i}" for i in range(2)]
    await vector_store.delete_vectors(
        collection_name=collection_name,
        ids=ids
    )
    
    # 验证结果
    collection = await vector_store.get_collection(collection_name)
    assert collection.count() == len(sample_vectors) - len(ids)

@pytest.mark.asyncio
async def test_get_collection_stats(
    vector_store,
    sample_documents,
    sample_vectors,
    sample_metadata
):
    """测试获取集合统计信息"""
    collection_name = "test_collection"
    metadata = {"description": "测试集合"}
    await vector_store.create_collection(collection_name, metadata)
    
    # 添加向量
    await vector_store.add_vectors(
        collection_name=collection_name,
        vectors=sample_vectors,
        documents=sample_documents,
        metadatas=sample_metadata
    )
    
    # 获取统计信息
    stats = await vector_store.get_collection_stats(collection_name)
    
    # 验证结果
    assert stats["name"] == collection_name
    assert stats["count"] == len(sample_vectors)
    assert stats["metadata"]["description"] == metadata["description"]

@pytest.mark.asyncio
async def test_reset(vector_store):
    """测试重置向量存储"""
    # 创建测试数据
    collection_name = "test_collection"
    await vector_store.create_collection(collection_name)
    
    # 重置向量存储
    await vector_store.reset()
    
    # 验证结果
    collections = await vector_store.list_collections()
    assert len(collections) == 0 