import chromadb
from chromadb.config import Settings

def test_chroma_basic():
    # 创建客户端
    client = chromadb.Client(Settings(
        persist_directory="./data/chroma"
    ))
    
    # 创建或获取集合
    collection = client.get_or_create_collection(name="test_collection")
    
    # 添加一些测试数据
    collection.add(
        documents=["这是一个测试文档", "这是另一个测试文档"],
        metadatas=[{"source": "test1"}, {"source": "test2"}],
        ids=["doc1", "doc2"]
    )
    
    # 测试查询
    results = collection.query(
        query_texts=["测试文档"],
        n_results=2
    )
    
    print("查询结果：", results)
    
    # 验证结果
    assert len(results['documents'][0]) > 0, "查询应该返回结果"
    print("基本功能测试通过！")

if __name__ == "__main__":
    test_chroma_basic() 