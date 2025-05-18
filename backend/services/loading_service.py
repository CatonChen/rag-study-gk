from typing import List, Dict, Optional, Union, Any
import logging
from pathlib import Path
import shutil
import aiofiles
from fastapi import UploadFile
from pydantic import BaseModel
from datetime import datetime
import mimetypes
import magic
import hashlib
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileMetadata(BaseModel):
    """文件元数据模型"""
    filename: str
    file_type: str
    size: int
    upload_time: str
    original_path: str
    saved_path: str
    mime_type: Optional[str] = None
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None
    encoding: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None

class LoadingService:
    """文档加载服务"""
    
    def __init__(self, upload_dir: str = "01-loaded-docs"):
        self.upload_dir = Path(upload_dir)
        self.supported_extensions = {
            '.pdf', '.docx', '.txt', '.csv', '.json', 
            '.md', '.html', '.htm', '.rtf', '.odt'
        }
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.upload_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    async def load_file(
        self, 
        file: UploadFile,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> FileMetadata:
        """
        加载单个文件
        
        Args:
            file: 上传的文件对象
            custom_metadata: 自定义元数据
            
        Returns:
            FileMetadata: 文件元数据
        """
        try:
            # 检查文件类型
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.supported_extensions:
                raise ValueError(f"不支持的文件类型: {file_ext}")
            
            # 生成保存路径
            save_path = self.upload_dir / file.filename
            
            # 保存文件
            async with aiofiles.open(save_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # 计算文件哈希
            md5_hash = hashlib.md5(content).hexdigest()
            sha256_hash = hashlib.sha256(content).hexdigest()
            
            # 检测MIME类型
            mime_type = magic.from_buffer(content, mime=True)
            
            # 获取文件元数据
            metadata = FileMetadata(
                filename=file.filename,
                file_type=file_ext[1:],
                size=len(content),
                upload_time=datetime.now().isoformat(),
                original_path=str(file.filename),
                saved_path=str(save_path),
                mime_type=mime_type,
                hash_md5=md5_hash,
                hash_sha256=sha256_hash,
                custom_metadata=custom_metadata
            )
            
            # 保存元数据
            await self._save_metadata(metadata)
            
            logger.info(f"文件加载成功: {file.filename}")
            return metadata
            
        except Exception as e:
            logger.error(f"文件加载失败: {str(e)}")
            raise
    
    async def load_files(
        self, 
        files: List[UploadFile],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[FileMetadata]:
        """
        批量加载文件
        
        Args:
            files: 上传的文件对象列表
            custom_metadata: 自定义元数据
            
        Returns:
            List[FileMetadata]: 文件元数据列表
        """
        results = []
        for file in files:
            try:
                metadata = await self.load_file(file, custom_metadata)
                results.append(metadata)
            except Exception as e:
                logger.error(f"文件 {file.filename} 加载失败: {str(e)}")
                continue
        return results
    
    async def delete_file(self, filename: str) -> bool:
        """
        删除已加载的文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否删除成功
        """
        try:
            file_path = self.upload_dir / filename
            metadata_path = self.metadata_dir / f"{filename}.json"
            
            if file_path.exists():
                file_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
            logger.info(f"文件删除成功: {filename}")
            return True
        except Exception as e:
            logger.error(f"文件删除失败: {str(e)}")
            return False
    
    def list_files(self) -> List[FileMetadata]:
        """
        列出所有已加载的文件
        
        Returns:
            List[FileMetadata]: 文件元数据列表
        """
        try:
            files = []
            for file_path in self.upload_dir.glob("*"):
                if file_path.suffix.lower() in self.supported_extensions:
                    metadata_path = self.metadata_dir / f"{file_path.name}.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = FileMetadata(**json.load(f))
                            files.append(metadata)
            return files
        except Exception as e:
            logger.error(f"文件列表获取失败: {str(e)}")
            return []
    
    async def _save_metadata(self, metadata: FileMetadata) -> None:
        """
        保存文件元数据
        
        Args:
            metadata: 文件元数据
        """
        try:
            metadata_path = self.metadata_dir / f"{metadata.filename}.json"
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(metadata.json(indent=2))
        except Exception as e:
            logger.error(f"元数据保存失败: {str(e)}")
            raise
    
    async def get_file_metadata(self, filename: str) -> Optional[FileMetadata]:
        """
        获取文件元数据
        
        Args:
            filename: 文件名
            
        Returns:
            Optional[FileMetadata]: 文件元数据
        """
        try:
            metadata_path = self.metadata_dir / f"{filename}.json"
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    return FileMetadata(**json.loads(content))
            return None
        except Exception as e:
            logger.error(f"元数据获取失败: {str(e)}")
            return None 