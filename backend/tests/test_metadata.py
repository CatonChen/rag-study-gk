import asyncio
import sys
from pathlib import Path
import gc
import shutil
import json

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from services.metadata_service import MetadataService, MetadataType, MetadataStatus

async def test_metadata_service():
    # 创建测试目录
    test_dir = Path("test_metadata")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    try:
        # 初始化元数据服务
        metadata_service = MetadataService(metadata_dir=str(test_dir))
        
        # 测试创建元数据
        print("\n=== 测试创建元数据 ===")
        document_metadata = await metadata_service.create_metadata(
            metadata_type=MetadataType.DOCUMENT,
            data={
                "filename": "test.txt",
                "size": 1024,
                "format": "txt"
            },
            author="test_user",
            tags=["test", "document"],
            custom_fields={"category": "test"}
        )
        print(f"创建的元数据ID: {document_metadata.id}")
        print(f"元数据类型: {document_metadata.type}")
        print(f"元数据状态: {document_metadata.status}")
        print(f"元数据内容: {document_metadata.data}")
        print(f"标签: {document_metadata.tags}")
        print(f"自定义字段: {document_metadata.custom_fields}")
        
        # 测试获取元数据
        print("\n=== 测试获取元数据 ===")
        retrieved_metadata = await metadata_service.get_metadata(
            metadata_id=document_metadata.id,
            metadata_type=MetadataType.DOCUMENT
        )
        print(f"获取的元数据ID: {retrieved_metadata.id}")
        print(f"元数据内容: {retrieved_metadata.data}")
        
        # 测试更新元数据
        print("\n=== 测试更新元数据 ===")
        updated_metadata = await metadata_service.update_metadata(
            metadata_id=document_metadata.id,
            metadata_type=MetadataType.DOCUMENT,
            data={
                "size": 2048,
                "updated": True
            },
            author="test_user",
            comment="更新文件大小"
        )
        print(f"更新后的元数据版本: {updated_metadata.version}")
        print(f"更新后的元数据内容: {updated_metadata.data}")
        print(f"版本历史: {[v.version for v in updated_metadata.versions]}")
        
        # 测试搜索元数据
        print("\n=== 测试搜索元数据 ===")
        search_results = await metadata_service.search_metadata(
            metadata_type=MetadataType.DOCUMENT,
            query={"format": "txt"},
            status=MetadataStatus.ACTIVE
        )
        print(f"搜索结果数量: {len(search_results)}")
        for result in search_results:
            print(f"匹配的元数据ID: {result.id}")
            print(f"元数据内容: {result.data}")
        
        # 测试获取元数据历史
        print("\n=== 测试获取元数据历史 ===")
        history = await metadata_service.get_metadata_history(
            metadata_id=document_metadata.id,
            metadata_type=MetadataType.DOCUMENT
        )
        print(f"历史版本数量: {len(history)}")
        for version in history:
            print(f"版本 {version.version}:")
            print(f"  作者: {version.author}")
            print(f"  变更: {version.changes}")
            print(f"  说明: {version.comment}")
        
        # 测试恢复元数据版本
        print("\n=== 测试恢复元数据版本 ===")
        restored_metadata = await metadata_service.restore_metadata_version(
            metadata_id=document_metadata.id,
            metadata_type=MetadataType.DOCUMENT,
            version=1,
            author="test_user",
            comment="恢复到初始版本"
        )
        print(f"恢复后的元数据版本: {restored_metadata.version}")
        print(f"恢复后的元数据内容: {restored_metadata.data}")
        
        # 测试删除元数据
        print("\n=== 测试删除元数据 ===")
        delete_result = await metadata_service.delete_metadata(
            metadata_id=document_metadata.id,
            metadata_type=MetadataType.DOCUMENT,
            author="test_user",
            comment="测试删除"
        )
        print(f"删除结果: {delete_result}")
        
        # 验证删除后的状态
        deleted_metadata = await metadata_service.get_metadata(
            metadata_id=document_metadata.id,
            metadata_type=MetadataType.DOCUMENT
        )
        print(f"删除后的状态: {deleted_metadata.status if deleted_metadata else 'Not found'}")
        
    finally:
        # 清理测试目录
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    asyncio.run(test_metadata_service()) 