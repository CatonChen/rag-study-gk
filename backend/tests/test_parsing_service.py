import pytest
import asyncio
from pathlib import Path
import tempfile
import os
from datetime import datetime
from services.parsing_service import (
    ParsingService,
    ParsingStrategy,
    ProcessingStatus,
    BatchProcessingConfig,
    ParsingError,
    FileNotFoundError,
    UnsupportedFileTypeError,
    FileReadError,
    FileParseError,
    MetadataExtractionError,
    ChunkProcessingError,
    BatchProcessingError,
    ProcessingTimeoutError,
    ProcessingCancelledError,
    DocumentMetadata,
    ParsedContent
)

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

@pytest.fixture
def parsing_service():
    """创建解析服务实例"""
    config = BatchProcessingConfig(
        max_concurrent_tasks=2,
        chunk_size=1024,  # 1KB
        retry_count=2,
        retry_delay=0.1
    )
    return ParsingService(config)

@pytest.fixture
def sample_text_file():
    """创建示例文本文件"""
    content = "这是一个测试文件。\n包含多行文本。\n用于测试解析功能。"
    file_path = TEST_DATA_DIR / "test.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    yield file_path
    file_path.unlink()

@pytest.fixture
def sample_markdown_file():
    """创建示例Markdown文件"""
    content = """# 标题1
## 标题2
- 列表项1
- 列表项2

> 引用文本

**粗体文本**
"""
    file_path = TEST_DATA_DIR / "test.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    yield file_path
    file_path.unlink()

@pytest.fixture
def sample_html_file():
    """创建示例HTML文件"""
    content = """<!DOCTYPE html>
<html>
<head>
    <title>测试页面</title>
</head>
<body>
    <h1>标题1</h1>
    <p>段落1</p>
    <table>
        <tr><td>单元格1</td><td>单元格2</td></tr>
    </table>
</body>
</html>"""
    file_path = TEST_DATA_DIR / "test.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    yield file_path
    file_path.unlink()

@pytest.mark.asyncio
async def test_parse_text_file(parsing_service, sample_text_file):
    """测试文本文件解析"""
    contents = await parsing_service.parse_document(
        sample_text_file,
        strategy=ParsingStrategy.ALL_TEXT
    )
    assert len(contents) == 1
    assert "测试文件" in contents[0].content
    assert contents[0].metadata["file_type"] == "txt"

@pytest.mark.asyncio
async def test_parse_markdown_file(parsing_service, sample_markdown_file):
    """测试Markdown文件解析"""
    contents = await parsing_service.parse_document(
        sample_markdown_file,
        strategy=ParsingStrategy.ALL_TEXT
    )
    assert len(contents) == 1
    assert "标题1" in contents[0].content
    assert "列表项1" in contents[0].content
    assert contents[0].metadata["file_type"] == "md"

@pytest.mark.asyncio
async def test_parse_html_file(parsing_service, sample_html_file):
    """测试HTML文件解析"""
    contents = await parsing_service.parse_document(
        sample_html_file,
        strategy=ParsingStrategy.TEXT_AND_TABLES
    )
    assert len(contents) == 1
    assert "标题1" in contents[0].content
    assert contents[0].tables is not None
    assert len(contents[0].tables) > 0
    assert contents[0].metadata["file_type"] == "html"

@pytest.mark.asyncio
async def test_file_not_found_error(parsing_service):
    """测试文件不存在错误"""
    with pytest.raises(FileNotFoundError):
        await parsing_service.parse_document("nonexistent.txt")

@pytest.mark.asyncio
async def test_unsupported_file_type_error(parsing_service):
    """测试不支持的文件类型错误"""
    with tempfile.NamedTemporaryFile(suffix=".xyz") as f:
        with pytest.raises(UnsupportedFileTypeError):
            await parsing_service.parse_document(f.name)

@pytest.mark.asyncio
async def test_batch_processing(parsing_service, sample_text_file, sample_markdown_file):
    """测试批量处理"""
    results = await parsing_service.process_batch([
        sample_text_file,
        sample_markdown_file
    ])
    
    assert len(results) == 2
    assert all(result.status == ProcessingStatus.COMPLETED for result in results.values())
    assert all(result.contents is not None for result in results.values())

@pytest.mark.asyncio
async def test_processing_status(parsing_service, sample_text_file):
    """测试处理状态查询"""
    # 开始处理
    task = asyncio.create_task(
        parsing_service.parse_document(sample_text_file)
    )
    
    # 查询状态
    status = await parsing_service.get_processing_status(str(sample_text_file))
    assert status is not None
    
    # 等待处理完成
    await task

@pytest.mark.asyncio
async def test_cancel_processing(parsing_service, sample_text_file):
    """测试取消处理"""
    # 开始处理
    task = asyncio.create_task(
        parsing_service.parse_document(sample_text_file)
    )
    
    # 取消处理
    success = await parsing_service.cancel_processing(str(sample_text_file))
    assert success
    
    # 检查状态
    status = await parsing_service.get_processing_status(str(sample_text_file))
    assert status.status == ProcessingStatus.FAILED
    assert "处理被取消" in status.error

@pytest.mark.asyncio
async def test_clear_processing_results(parsing_service, sample_text_file):
    """测试清理处理结果"""
    # 处理文件
    await parsing_service.parse_document(sample_text_file)
    
    # 清理结果
    await parsing_service.clear_processing_results([str(sample_text_file)])
    
    # 验证结果已清理
    status = await parsing_service.get_processing_status(str(sample_text_file))
    assert status is None

@pytest.mark.asyncio
async def test_error_handling(parsing_service):
    """测试错误处理"""
    # 创建无效文件
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Invalid content")
        invalid_file = f.name
    
    try:
        # 测试文件读取错误
        with pytest.raises(FileReadError):
            await parsing_service.parse_document(invalid_file)
        
        # 测试批量处理错误
        results = await parsing_service.process_batch([invalid_file])
        assert results[invalid_file].status == ProcessingStatus.FAILED
        assert "失败" in results[invalid_file].error
        
    finally:
        # 清理测试文件
        os.unlink(invalid_file)

@pytest.mark.asyncio
async def test_metadata_extraction(parsing_service, sample_text_file):
    """测试元数据提取"""
    contents = await parsing_service.parse_document(sample_text_file)
    metadata = contents[0].metadata
    
    assert metadata["filename"] == sample_text_file.name
    assert metadata["file_type"] == "txt"
    assert metadata["language"] is not None
    assert metadata["word_count"] > 0

@pytest.mark.asyncio
async def test_chunk_processing(parsing_service):
    """测试分片处理"""
    # 创建大文件
    large_file = TEST_DATA_DIR / "large.txt"
    try:
        with open(large_file, "w", encoding="utf-8") as f:
            for i in range(1000):
                f.write(f"这是第 {i} 行测试文本。\n")
        
        contents = await parsing_service.parse_document(large_file)
        assert len(contents) > 1  # 应该被分片
        assert all(c.chunk_index is not None for c in contents)
        assert all(c.total_chunks is not None for c in contents)
        
    finally:
        large_file.unlink()

def teardown_module(module):
    """清理测试数据目录"""
    if TEST_DATA_DIR.exists():
        for file in TEST_DATA_DIR.iterdir():
            file.unlink()
        TEST_DATA_DIR.rmdir() 