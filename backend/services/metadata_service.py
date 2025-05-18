from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from pathlib import Path
import logging
from pydantic import BaseModel, Field
import hashlib
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataType(str, Enum):
    """元数据类型枚举"""
    DOCUMENT = "document"      # 文档元数据
    CHUNK = "chunk"           # 分块元数据
    SYSTEM = "system"         # 系统元数据
    CUSTOM = "custom"         # 自定义元数据

class MetadataStatus(str, Enum):
    """元数据状态枚举"""
    ACTIVE = "active"         # 活跃
    ARCHIVED = "archived"     # 归档
    DELETED = "deleted"       # 删除
    PENDING = "pending"       # 待处理

class MetadataVersion(BaseModel):
    """元数据版本模型"""
    version: int = Field(..., description="版本号")
    timestamp: datetime = Field(default_factory=datetime.now, description="版本时间戳")
    author: str = Field(..., description="版本作者")
    changes: Dict[str, Any] = Field(..., description="变更内容")
    comment: Optional[str] = Field(None, description="版本说明")

class Metadata(BaseModel):
    """元数据基础模型"""
    id: str = Field(..., description="元数据ID")
    type: MetadataType = Field(..., description="元数据类型")
    status: MetadataStatus = Field(default=MetadataStatus.ACTIVE, description="元数据状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    version: int = Field(default=1, description="当前版本号")
    versions: List[MetadataVersion] = Field(default_factory=list, description="版本历史")
    data: Dict[str, Any] = Field(default_factory=dict, description="元数据内容")
    tags: List[str] = Field(default_factory=list, description="标签")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="自定义字段")

class MetadataService:
    """元数据管理服务"""
    
    def __init__(self, metadata_dir: str = "metadata"):
        """
        初始化元数据服务
        
        Args:
            metadata_dir: 元数据存储目录
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._create_index()
    
    def _create_index(self):
        """创建元数据索引"""
        self.index_file = self.metadata_dir / "index.json"
        if not self.index_file.exists():
            self.index = {
                "documents": {},
                "chunks": {},
                "system": {},
                "custom": {}
            }
            self._save_index()
        else:
            self._load_index()
    
    def _save_index(self):
        """保存索引"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
    
    def _load_index(self):
        """加载索引"""
        with open(self.index_file, "r", encoding="utf-8") as f:
            self.index = json.load(f)
    
    def _generate_id(self, content: Dict[str, Any]) -> str:
        """
        生成元数据ID
        
        Args:
            content: 元数据内容
            
        Returns:
            str: 元数据ID
        """
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _get_metadata_path(self, metadata_id: str, metadata_type: MetadataType) -> Path:
        """
        获取元数据文件路径
        
        Args:
            metadata_id: 元数据ID
            metadata_type: 元数据类型
            
        Returns:
            Path: 元数据文件路径
        """
        return self.metadata_dir / metadata_type.value / f"{metadata_id}.json"
    
    async def create_metadata(
        self,
        metadata_type: MetadataType,
        data: Dict[str, Any],
        author: str,
        tags: Optional[List[str]] = None,
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> Metadata:
        """
        创建元数据
        
        Args:
            metadata_type: 元数据类型
            data: 元数据内容
            author: 作者
            tags: 标签
            custom_fields: 自定义字段
            
        Returns:
            Metadata: 创建的元数据
        """
        try:
            # 生成元数据ID
            metadata_id = self._generate_id(data)
            
            # 创建元数据对象
            metadata = Metadata(
                id=metadata_id,
                type=metadata_type,
                data=data,
                tags=tags or [],
                custom_fields=custom_fields or {},
                versions=[
                    MetadataVersion(
                        version=1,
                        author=author,
                        changes=data,
                        comment="Initial version"
                    )
                ]
            )
            
            # 保存元数据
            metadata_path = self._get_metadata_path(metadata_id, metadata_type)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.dict(), f, ensure_ascii=False, indent=2)
            
            # 更新索引
            self.index[metadata_type.value][metadata_id] = {
                "path": str(metadata_path),
                "status": metadata.status,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat()
            }
            self._save_index()
            
            return metadata
            
        except Exception as e:
            logger.error(f"创建元数据失败: {str(e)}")
            raise
    
    async def get_metadata(
        self,
        metadata_id: str,
        metadata_type: MetadataType
    ) -> Optional[Metadata]:
        """
        获取元数据
        
        Args:
            metadata_id: 元数据ID
            metadata_type: 元数据类型
            
        Returns:
            Optional[Metadata]: 元数据对象
        """
        try:
            metadata_path = self._get_metadata_path(metadata_id, metadata_type)
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            
            return Metadata(**metadata_dict)
            
        except Exception as e:
            logger.error(f"获取元数据失败: {str(e)}")
            raise
    
    async def update_metadata(
        self,
        metadata_id: str,
        metadata_type: MetadataType,
        data: Dict[str, Any],
        author: str,
        comment: Optional[str] = None
    ) -> Optional[Metadata]:
        """
        更新元数据
        
        Args:
            metadata_id: 元数据ID
            metadata_type: 元数据类型
            data: 更新内容
            author: 作者
            comment: 更新说明
            
        Returns:
            Optional[Metadata]: 更新后的元数据
        """
        try:
            metadata = await self.get_metadata(metadata_id, metadata_type)
            if not metadata:
                return None
            
            # 创建新版本
            new_version = MetadataVersion(
                version=metadata.version + 1,
                author=author,
                changes=data,
                comment=comment
            )
            
            # 更新元数据
            metadata.version = new_version.version
            metadata.updated_at = datetime.now()
            metadata.versions.append(new_version)
            metadata.data.update(data)
            
            # 保存更新
            metadata_path = self._get_metadata_path(metadata_id, metadata_type)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.dict(), f, ensure_ascii=False, indent=2)
            
            # 更新索引
            self.index[metadata_type.value][metadata_id]["updated_at"] = metadata.updated_at.isoformat()
            self._save_index()
            
            return metadata
            
        except Exception as e:
            logger.error(f"更新元数据失败: {str(e)}")
            raise
    
    async def delete_metadata(
        self,
        metadata_id: str,
        metadata_type: MetadataType,
        author: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        删除元数据
        
        Args:
            metadata_id: 元数据ID
            metadata_type: 元数据类型
            author: 作者
            comment: 删除说明
            
        Returns:
            bool: 是否删除成功
        """
        try:
            metadata = await self.get_metadata(metadata_id, metadata_type)
            if not metadata:
                return False
            
            # 更新状态为已删除
            metadata.status = MetadataStatus.DELETED
            metadata.updated_at = datetime.now()
            
            # 添加删除版本
            new_version = MetadataVersion(
                version=metadata.version + 1,
                author=author,
                changes={"status": MetadataStatus.DELETED},
                comment=comment or "Metadata deleted"
            )
            metadata.versions.append(new_version)
            
            # 保存更新
            metadata_path = self._get_metadata_path(metadata_id, metadata_type)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.dict(), f, ensure_ascii=False, indent=2)
            
            # 更新索引
            self.index[metadata_type.value][metadata_id]["status"] = MetadataStatus.DELETED
            self.index[metadata_type.value][metadata_id]["updated_at"] = metadata.updated_at.isoformat()
            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"删除元数据失败: {str(e)}")
            raise
    
    async def search_metadata(
        self,
        metadata_type: MetadataType,
        query: Dict[str, Any],
        status: Optional[MetadataStatus] = None
    ) -> List[Metadata]:
        """
        搜索元数据
        
        Args:
            metadata_type: 元数据类型
            query: 搜索条件
            status: 元数据状态
            
        Returns:
            List[Metadata]: 匹配的元数据列表
        """
        try:
            results = []
            for metadata_id in self.index[metadata_type.value]:
                metadata = await self.get_metadata(metadata_id, metadata_type)
                if not metadata:
                    continue
                
                # 检查状态
                if status and metadata.status != status:
                    continue
                
                # 检查查询条件
                match = True
                for key, value in query.items():
                    if key not in metadata.data or metadata.data[key] != value:
                        match = False
                        break
                
                if match:
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索元数据失败: {str(e)}")
            raise
    
    async def get_metadata_history(
        self,
        metadata_id: str,
        metadata_type: MetadataType
    ) -> List[MetadataVersion]:
        """
        获取元数据历史版本
        
        Args:
            metadata_id: 元数据ID
            metadata_type: 元数据类型
            
        Returns:
            List[MetadataVersion]: 版本历史列表
        """
        try:
            metadata = await self.get_metadata(metadata_id, metadata_type)
            if not metadata:
                return []
            
            return metadata.versions
            
        except Exception as e:
            logger.error(f"获取元数据历史失败: {str(e)}")
            raise
    
    async def restore_metadata_version(
        self,
        metadata_id: str,
        metadata_type: MetadataType,
        version: int,
        author: str,
        comment: Optional[str] = None
    ) -> Optional[Metadata]:
        """
        恢复元数据到指定版本
        
        Args:
            metadata_id: 元数据ID
            metadata_type: 元数据类型
            version: 要恢复的版本号
            author: 作者
            comment: 恢复说明
            
        Returns:
            Optional[Metadata]: 恢复后的元数据
        """
        try:
            metadata = await self.get_metadata(metadata_id, metadata_type)
            if not metadata:
                return None
            
            # 查找指定版本
            target_version = None
            for v in metadata.versions:
                if v.version == version:
                    target_version = v
                    break
            
            if not target_version:
                return None
            
            # 创建新版本
            new_version = MetadataVersion(
                version=metadata.version + 1,
                author=author,
                changes=target_version.changes,
                comment=comment or f"Restored to version {version}"
            )
            
            # 更新元数据
            metadata.version = new_version.version
            metadata.updated_at = datetime.now()
            metadata.versions.append(new_version)
            metadata.data = target_version.changes
            
            # 保存更新
            metadata_path = self._get_metadata_path(metadata_id, metadata_type)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.dict(), f, ensure_ascii=False, indent=2)
            
            # 更新索引
            self.index[metadata_type.value][metadata_id]["updated_at"] = metadata.updated_at.isoformat()
            self._save_index()
            
            return metadata
            
        except Exception as e:
            logger.error(f"恢复元数据版本失败: {str(e)}")
            raise 