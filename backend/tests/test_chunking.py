import asyncio
import sys
from pathlib import Path
import gc

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from services.chunking_service import ChunkingService, ChunkingStrategy

async def test_chunking_service():
    # 初始化分块服务
    chunking_service = ChunkingService()
    
    # 测试文本
    test_text = """
    # 测试文档
    
    ## 第一部分
    这是第一段文本。这是一个测试段落。
    这里包含多个句子。这是第二个句子。
    
    ## 第二部分
    这是第二段文本。这是另一个测试段落。
    这里也包含多个句子。这是另一个句子。
    
    ## 代码示例
    def test_function():
        print("这是一个测试函数")
        return True
    
    ## 表格示例
    | 列1 | 列2 | 列3 |
    |-----|-----|-----|
    | 数据1 | 数据2 | 数据3 |
    | 数据4 | 数据5 | 数据6 |
    """
    
    # 测试按页面分块
    print("\n=== 按页面分块 ===")
    page_map = [
        {"page": 1, "text": "这是第一页的内容。"},
        {"page": 2, "text": "这是第二页的内容。"}
    ]
    page_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_PAGES,
        source_file="test.txt",
        page_map=page_map,
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {page_chunks.total_chunks}")
    print(f"总页数: {page_chunks.total_pages}")
    for chunk in page_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"页码: {chunk.metadata.page_number}")
        print(f"内容: {chunk.content}")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del page_chunks
    gc.collect()
    
    # 测试按段落分块
    print("\n=== 按段落分块 ===")
    paragraph_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_PARAGRAPH,
        chunk_size=50,
        overlap=10,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {paragraph_chunks.total_chunks}")
    for chunk in paragraph_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"内容: {chunk.content[:30]}...")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del paragraph_chunks
    gc.collect()
    
    # 测试按句子分块
    print("\n=== 按句子分块 ===")
    sentence_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_SENTENCE,
        chunk_size=30,
        overlap=5,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {sentence_chunks.total_chunks}")
    for chunk in sentence_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"内容: {chunk.content[:30]}...")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del sentence_chunks
    gc.collect()
    
    # 测试按固定大小分块
    print("\n=== 按固定大小分块 ===")
    fixed_size_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_FIXED_SIZE,
        chunk_size=20,
        overlap=5,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {fixed_size_chunks.total_chunks}")
    for chunk in fixed_size_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"内容: {chunk.content}")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del fixed_size_chunks
    gc.collect()
    
    # 测试语义分块
    print("\n=== 语义分块 ===")
    semantic_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_SEMANTIC,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {semantic_chunks.total_chunks}")
    print(f"平均语义得分: {semantic_chunks.avg_semantic_score:.2f}")
    for chunk in semantic_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"内容: {chunk.content[:30]}...")
        print(f"语义得分: {chunk.metadata.semantic_score:.2f}")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del semantic_chunks
    gc.collect()
    
    # 测试标题分块
    print("\n=== 标题分块 ===")
    title_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_TITLE,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {title_chunks.total_chunks}")
    for chunk in title_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"标题级别: {chunk.metadata.title_level}")
        print(f"内容: {chunk.content[:30]}...")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del title_chunks
    gc.collect()
    
    # 测试混合分块
    print("\n=== 混合分块 ===")
    hybrid_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_HYBRID,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {hybrid_chunks.total_chunks}")
    for chunk in hybrid_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"质量得分: {chunk.metadata.quality_score:.2f}")
        print(f"内容: {chunk.content[:30]}...")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del hybrid_chunks
    gc.collect()
    
    # 测试自适应分块
    print("\n=== 自适应分块 ===")
    adaptive_chunks = await chunking_service.chunk_text(
        text=test_text,
        strategy=ChunkingStrategy.BY_ADAPTIVE,
        source_file="test.txt",
        metadata={"loading_method": "test"}
    )
    print(f"总块数: {adaptive_chunks.total_chunks}")
    print(f"内容类型分布: {adaptive_chunks.content_types}")
    for chunk in adaptive_chunks.chunks:
        print(f"\n分块ID: {chunk.metadata.chunk_id}")
        print(f"内容类型: {chunk.metadata.content_type}")
        print(f"质量得分: {chunk.metadata.quality_score:.2f}")
        print(f"内容: {chunk.content[:30]}...")
        print(f"词数: {chunk.metadata.word_count}")
    
    # 清理内存
    del adaptive_chunks
    gc.collect()

if __name__ == "__main__":
    asyncio.run(test_chunking_service()) 