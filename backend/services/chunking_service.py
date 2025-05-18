from typing import List, Dict, Optional, Union, Any
from enum import Enum
import logging
from pathlib import Path
import re
from datetime import datetime
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingStrategy(str, Enum):
    """分块策略枚举"""
    # 现有策略
    BY_PAGES = "by_pages"
    BY_PARAGRAPH = "by_paragraph"
    BY_SENTENCE = "by_sentence"
    BY_FIXED_SIZE = "by_fixed_size"
    
    # 新增策略
    BY_SEMANTIC = "by_semantic"        # 语义分块
    BY_TITLE = "by_title"              # 标题分块
    BY_HYBRID = "by_hybrid"            # 混合分块
    BY_ADAPTIVE = "by_adaptive"        # 自适应分块

class ChunkingConfig(BaseModel):
    """分块配置模型"""
    # 基础配置
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    # 语义分块配置
    semantic_threshold: float = 0.8
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 标题分块配置
    title_patterns: List[str] = ["^#+", "^[A-Z][^a-z]"]
    max_title_level: int = 3
    
    # 混合分块配置
    strategy_weights: Dict[str, float] = {
        "semantic": 0.4,
        "title": 0.3,
        "paragraph": 0.3
    }
    
    # 自适应分块配置
    adaptive_threshold: float = 0.7
    content_type_weights: Dict[str, float] = {
        "text": 1.0,
        "code": 0.8,
        "table": 0.6
    }

class ChunkMetadata(BaseModel):
    """分块元数据模型"""
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    word_count: int
    start_index: int
    end_index: int
    strategy: str
    overlap: int = 0
    # 新增元数据字段
    semantic_score: Optional[float] = None
    title_level: Optional[int] = None
    content_type: Optional[str] = None
    quality_score: Optional[float] = None

class TextChunk(BaseModel):
    """文本分块模型"""
    content: str
    metadata: ChunkMetadata

class DocumentData(BaseModel):
    """文档数据结构"""
    filename: str
    total_chunks: int
    total_pages: int
    loading_method: str
    chunking_method: str
    timestamp: str
    chunks: List[TextChunk]
    # 新增文档级元数据
    avg_chunk_size: float
    avg_semantic_score: Optional[float] = None
    structure_complexity: Optional[float] = None
    content_types: Dict[str, int] = {}

class ChunkingService:
    """文档分块服务"""
    
    def __init__(self, chunk_dir: str = "01-chunked-docs"):
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.max_text_length = 1000000  # 设置最大文本长度限制
        self.embedding_model = None  # 延迟加载嵌入模型
        self.config = ChunkingConfig()
    
    def _load_embedding_model(self):
        """延迟加载嵌入模型"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        return self.embedding_model
    
    async def chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_PARAGRAPH,
        chunk_size: int = 1000,
        overlap: int = 200,
        source_file: str = "unknown",
        page_map: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentData:
        """
        对文本进行分块
        
        Args:
            text: 要分块的文本
            strategy: 分块策略
            chunk_size: 分块大小
            overlap: 重叠大小
            source_file: 源文件名
            page_map: 页面映射列表
            metadata: 文档元数据
            
        Returns:
            DocumentData: 分块结果
        """
        try:
            # 检查文本长度
            if len(text) > self.max_text_length:
                raise ValueError(f"文本长度超过限制: {len(text)} > {self.max_text_length}")
            
            # 验证参数
            if chunk_size <= 0:
                raise ValueError("chunk_size 必须大于 0")
            if overlap < 0:
                raise ValueError("overlap 不能为负数")
            if overlap >= chunk_size:
                raise ValueError("overlap 必须小于 chunk_size")
            
            # 更新配置
            self.config.chunk_size = chunk_size
            self.config.overlap = overlap
            
            chunks = []
            if strategy == ChunkingStrategy.BY_PAGES:
                if not page_map:
                    raise ValueError("按页面分块需要提供 page_map")
                chunks = await self._chunk_by_pages(page_map, source_file)
            elif strategy == ChunkingStrategy.BY_PARAGRAPH:
                chunks = await self._chunk_by_paragraph(text, chunk_size, overlap, source_file)
            elif strategy == ChunkingStrategy.BY_SENTENCE:
                chunks = await self._chunk_by_sentence(text, chunk_size, overlap, source_file)
            elif strategy == ChunkingStrategy.BY_FIXED_SIZE:
                chunks = await self._chunk_by_fixed_size(text, chunk_size, overlap, source_file)
            elif strategy == ChunkingStrategy.BY_SEMANTIC:
                chunks = await self._chunk_by_semantic(text, source_file)
            elif strategy == ChunkingStrategy.BY_TITLE:
                chunks = await self._chunk_by_title(text, source_file)
            elif strategy == ChunkingStrategy.BY_HYBRID:
                chunks = await self._chunk_by_hybrid(text, source_file)
            elif strategy == ChunkingStrategy.BY_ADAPTIVE:
                chunks = await self._chunk_by_adaptive(text, source_file)
            else:
                raise ValueError(f"不支持的分块策略: {strategy}")
            
            # 计算文档级统计信息
            avg_chunk_size = sum(len(chunk.content) for chunk in chunks) / len(chunks)
            avg_semantic_score = sum(chunk.metadata.semantic_score or 0 for chunk in chunks) / len(chunks)
            content_types = {}
            for chunk in chunks:
                if chunk.metadata.content_type:
                    content_types[chunk.metadata.content_type] = content_types.get(chunk.metadata.content_type, 0) + 1
            
            # 创建标准化的文档数据结构
            return DocumentData(
                filename=source_file,
                total_chunks=len(chunks),
                total_pages=len(page_map) if page_map else 1,
                loading_method=metadata.get("loading_method", "unknown") if metadata else "unknown",
                chunking_method=strategy,
                timestamp=datetime.now().isoformat(),
                chunks=chunks,
                avg_chunk_size=avg_chunk_size,
                avg_semantic_score=avg_semantic_score,
                content_types=content_types
            )
            
        except Exception as e:
            logger.error(f"文本分块失败: {str(e)}")
            raise
    
    async def _chunk_by_pages(
        self,
        page_map: List[Dict[str, Any]],
        source_file: str
    ) -> List[TextChunk]:
        """按页面分块"""
        try:
            chunks = []
            for page_data in page_map:
                chunk = TextChunk(
                    content=page_data['text'],
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_page_{page_data['page']}",
                        source_file=source_file,
                        page_number=page_data['page'],
                        page_range=str(page_data['page']),
                        word_count=len(page_data['text'].split()),
                        start_index=0,
                        end_index=len(page_data['text']),
                        strategy=ChunkingStrategy.BY_PAGES
                    )
                )
                chunks.append(chunk)
            return chunks
        except Exception as e:
            logger.error(f"按页面分块失败: {str(e)}")
            raise
    
    async def _chunk_by_paragraph(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        source_file: str
    ) -> List[TextChunk]:
        """按段落分块"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
            )
            texts = splitter.split_text(text)
            
            chunks = []
            for i, content in enumerate(texts):
                chunks.append(TextChunk(
                    content=content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_chunk_{i}",
                        source_file=source_file,
                        word_count=len(content.split()),
                        start_index=text.find(content),
                        end_index=text.find(content) + len(content),
                        strategy=ChunkingStrategy.BY_PARAGRAPH,
                        overlap=overlap
                    )
                ))
            return chunks
        except Exception as e:
            logger.error(f"按段落分块失败: {str(e)}")
            raise
    
    async def _chunk_by_sentence(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        source_file: str
    ) -> List[TextChunk]:
        """按句子分块"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["。", "！", "？", ".", "!", "?", "\n", " "]
            )
            texts = splitter.split_text(text)
            
            chunks = []
            for i, content in enumerate(texts):
                chunks.append(TextChunk(
                    content=content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_chunk_{i}",
                        source_file=source_file,
                        word_count=len(content.split()),
                        start_index=text.find(content),
                        end_index=text.find(content) + len(content),
                        strategy=ChunkingStrategy.BY_SENTENCE,
                        overlap=overlap
                    )
                ))
            return chunks
        except Exception as e:
            logger.error(f"按句子分块失败: {str(e)}")
            raise
    
    async def _chunk_by_fixed_size(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        source_file: str
    ) -> List[TextChunk]:
        """按固定大小分块"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n", "。", "！", "？", ".", "!", "?", " "]
            )
            texts = splitter.split_text(text)
            
            chunks = []
            for i, content in enumerate(texts):
                chunks.append(TextChunk(
                    content=content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_chunk_{i}",
                        source_file=source_file,
                        word_count=len(content.split()),
                        start_index=text.find(content),
                        end_index=text.find(content) + len(content),
                        strategy=ChunkingStrategy.BY_FIXED_SIZE,
                        overlap=overlap
                    )
                ))
            return chunks
        except Exception as e:
            logger.error(f"按固定大小分块失败: {str(e)}")
            raise 
    
    async def _chunk_by_semantic(
        self,
        text: str,
        source_file: str
    ) -> List[TextChunk]:
        """
        按语义分块
        
        Args:
            text: 要分块的文本
            source_file: 源文件名
            
        Returns:
            List[TextChunk]: 分块结果
        """
        try:
            # 首先按句子分割文本
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.overlap,
                separators=["。", "！", "？", ".", "!", "?", "\n", " "]
            )
            sentences = splitter.split_text(text)
            
            # 获取嵌入模型
            model = self._load_embedding_model()
            
            # 计算句子嵌入
            embeddings = model.encode(sentences)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            # 基于相似度进行分块
            chunks = []
            current_chunk = []
            current_embedding = None
            
            for i, sentence in enumerate(sentences):
                if not current_chunk:
                    current_chunk.append(sentence)
                    current_embedding = embeddings[i]
                    continue
                
                # 计算当前句子与当前块的相似度
                similarity = cosine_similarity(
                    [current_embedding],
                    [embeddings[i]]
                )[0][0]
                
                if similarity >= self.config.semantic_threshold:
                    # 如果相似度高，加入当前块
                    current_chunk.append(sentence)
                    # 更新当前块的嵌入（使用平均嵌入）
                    current_embedding = np.mean(
                        [embeddings[j] for j in range(i - len(current_chunk) + 1, i + 1)],
                        axis=0
                    )
                else:
                    # 如果相似度低，创建新块
                    chunk_content = " ".join(current_chunk)
                    chunks.append(TextChunk(
                        content=chunk_content,
                        metadata=ChunkMetadata(
                            chunk_id=f"{source_file}_semantic_{len(chunks)}",
                            source_file=source_file,
                            word_count=len(chunk_content.split()),
                            start_index=text.find(chunk_content),
                            end_index=text.find(chunk_content) + len(chunk_content),
                            strategy=ChunkingStrategy.BY_SEMANTIC,
                            semantic_score=float(similarity),
                            content_type="text"
                        )
                    ))
                    current_chunk = [sentence]
                    current_embedding = embeddings[i]
            
            # 处理最后一个块
            if current_chunk:
                chunk_content = " ".join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_semantic_{len(chunks)}",
                        source_file=source_file,
                        word_count=len(chunk_content.split()),
                        start_index=text.find(chunk_content),
                        end_index=text.find(chunk_content) + len(chunk_content),
                        strategy=ChunkingStrategy.BY_SEMANTIC,
                        semantic_score=1.0,  # 最后一个块的相似度设为1.0
                        content_type="text"
                    )
                ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"语义分块失败: {str(e)}")
            raise 
    
    async def _chunk_by_title(
        self,
        text: str,
        source_file: str
    ) -> List[TextChunk]:
        """
        按标题分块
        
        Args:
            text: 要分块的文本
            source_file: 源文件名
            
        Returns:
            List[TextChunk]: 分块结果
        """
        try:
            # 按行分割文本
            lines = text.split('\n')
            
            # 识别标题和内容
            chunks = []
            current_title = None
            current_content = []
            current_level = 0
            
            for line in lines:
                # 检查是否是标题
                is_title = False
                title_level = 0
                
                for pattern in self.config.title_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        is_title = True
                        # 计算标题级别
                        if pattern == "^#+":
                            title_level = len(match.group(0))
                        else:
                            title_level = 1
                        break
                
                if is_title:
                    # 如果已有内容，保存当前块
                    if current_content:
                        chunk_content = '\n'.join(current_content)
                        chunks.append(TextChunk(
                            content=chunk_content,
                            metadata=ChunkMetadata(
                                chunk_id=f"{source_file}_title_{len(chunks)}",
                                source_file=source_file,
                                word_count=len(chunk_content.split()),
                                start_index=text.find(chunk_content),
                                end_index=text.find(chunk_content) + len(chunk_content),
                                strategy=ChunkingStrategy.BY_TITLE,
                                title_level=current_level,
                                content_type="text"
                            )
                        ))
                    
                    # 开始新的块
                    current_title = line.strip()
                    current_content = [current_title]
                    current_level = title_level
                else:
                    # 添加内容到当前块
                    if current_content or line.strip():  # 忽略空行
                        current_content.append(line)
            
            # 处理最后一个块
            if current_content:
                chunk_content = '\n'.join(current_content)
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_title_{len(chunks)}",
                        source_file=source_file,
                        word_count=len(chunk_content.split()),
                        start_index=text.find(chunk_content),
                        end_index=text.find(chunk_content) + len(chunk_content),
                        strategy=ChunkingStrategy.BY_TITLE,
                        title_level=current_level,
                        content_type="text"
                    )
                ))
            
            # 如果没有找到标题，使用默认分块
            if not chunks:
                return await self._chunk_by_paragraph(text, self.config.chunk_size, self.config.overlap, source_file)
            
            return chunks
            
        except Exception as e:
            logger.error(f"标题分块失败: {str(e)}")
            raise 
    
    async def _chunk_by_hybrid(
        self,
        text: str,
        source_file: str
    ) -> List[TextChunk]:
        """
        混合分块策略
        
        Args:
            text: 要分块的文本
            source_file: 源文件名
            
        Returns:
            List[TextChunk]: 分块结果
        """
        try:
            # 获取不同策略的分块结果
            semantic_chunks = await self._chunk_by_semantic(text, source_file)
            title_chunks = await self._chunk_by_title(text, source_file)
            paragraph_chunks = await self._chunk_by_paragraph(
                text,
                self.config.chunk_size,
                self.config.overlap,
                source_file
            )
            
            # 计算每个策略的权重
            weights = self.config.strategy_weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # 合并分块结果
            merged_chunks = []
            chunk_scores = {}  # 记录每个位置的分块得分
            
            # 处理语义分块
            for chunk in semantic_chunks:
                start = chunk.metadata.start_index
                end = chunk.metadata.end_index
                score = normalized_weights["semantic"] * (chunk.metadata.semantic_score or 0.5)
                for i in range(start, end):
                    chunk_scores[i] = chunk_scores.get(i, 0) + score
            
            # 处理标题分块
            for chunk in title_chunks:
                start = chunk.metadata.start_index
                end = chunk.metadata.end_index
                score = normalized_weights["title"] * (1.0 - (chunk.metadata.title_level or 1) / self.config.max_title_level)
                for i in range(start, end):
                    chunk_scores[i] = chunk_scores.get(i, 0) + score
            
            # 处理段落分块
            for chunk in paragraph_chunks:
                start = chunk.metadata.start_index
                end = chunk.metadata.end_index
                score = normalized_weights["paragraph"]
                for i in range(start, end):
                    chunk_scores[i] = chunk_scores.get(i, 0) + score
            
            # 根据得分确定最终分块
            current_chunk = []
            current_start = 0
            current_score = 0
            
            for i, char in enumerate(text):
                if i in chunk_scores:
                    score = chunk_scores[i]
                    if score > current_score:
                        # 如果当前得分更高，开始新块
                        if current_chunk:
                            chunk_content = ''.join(current_chunk)
                            merged_chunks.append(TextChunk(
                                content=chunk_content,
                                metadata=ChunkMetadata(
                                    chunk_id=f"{source_file}_hybrid_{len(merged_chunks)}",
                                    source_file=source_file,
                                    word_count=len(chunk_content.split()),
                                    start_index=current_start,
                                    end_index=i,
                                    strategy=ChunkingStrategy.BY_HYBRID,
                                    quality_score=current_score,
                                    content_type="text"
                                )
                            ))
                        current_chunk = [char]
                        current_start = i
                        current_score = score
                    else:
                        # 继续当前块
                        current_chunk.append(char)
                else:
                    # 继续当前块
                    current_chunk.append(char)
            
            # 处理最后一个块
            if current_chunk:
                chunk_content = ''.join(current_chunk)
                merged_chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_hybrid_{len(merged_chunks)}",
                        source_file=source_file,
                        word_count=len(chunk_content.split()),
                        start_index=current_start,
                        end_index=len(text),
                        strategy=ChunkingStrategy.BY_HYBRID,
                        quality_score=current_score,
                        content_type="text"
                    )
                ))
            
            return merged_chunks
            
        except Exception as e:
            logger.error(f"混合分块失败: {str(e)}")
            raise 
    
    async def _chunk_by_adaptive(
        self,
        text: str,
        source_file: str
    ) -> List[TextChunk]:
        """
        自适应分块策略
        
        Args:
            text: 要分块的文本
            source_file: 源文件名
            
        Returns:
            List[TextChunk]: 分块结果
        """
        try:
            # 分析文本特征
            lines = text.split('\n')
            content_types = self._analyze_content_types(lines)
            
            # 根据内容类型调整分块参数
            chunks = []
            current_chunk = []
            current_type = None
            current_start = 0
            
            for i, line in enumerate(lines):
                line_type = self._detect_content_type(line)
                
                if not current_chunk:
                    # 开始新块
                    current_chunk.append(line)
                    current_type = line_type
                    current_start = text.find(line)
                elif line_type == current_type:
                    # 继续当前块
                    current_chunk.append(line)
                else:
                    # 保存当前块
                    chunk_content = '\n'.join(current_chunk)
                    chunk_size = len(chunk_content)
                    
                    # 根据内容类型调整分块大小
                    adjusted_size = int(self.config.chunk_size * 
                                     self.config.content_type_weights.get(current_type, 1.0))
                    
                    if chunk_size > adjusted_size:
                        # 如果块太大，进行子分块
                        sub_chunks = await self._sub_chunk_by_type(
                            chunk_content,
                            current_type,
                            adjusted_size,
                            source_file,
                            current_start
                        )
                        chunks.extend(sub_chunks)
                    else:
                        # 直接添加块
                        chunks.append(TextChunk(
                            content=chunk_content,
                            metadata=ChunkMetadata(
                                chunk_id=f"{source_file}_adaptive_{len(chunks)}",
                                source_file=source_file,
                                word_count=len(chunk_content.split()),
                                start_index=current_start,
                                end_index=text.find(chunk_content) + len(chunk_content),
                                strategy=ChunkingStrategy.BY_ADAPTIVE,
                                content_type=current_type,
                                quality_score=self._calculate_quality_score(chunk_content, current_type)
                            )
                        ))
                    
                    # 开始新块
                    current_chunk = [line]
                    current_type = line_type
                    current_start = text.find(line)
            
            # 处理最后一个块
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunk_size = len(chunk_content)
                
                # 根据内容类型调整分块大小
                adjusted_size = int(self.config.chunk_size * 
                                 self.config.content_type_weights.get(current_type, 1.0))
                
                if chunk_size > adjusted_size:
                    # 如果块太大，进行子分块
                    sub_chunks = await self._sub_chunk_by_type(
                        chunk_content,
                        current_type,
                        adjusted_size,
                        source_file,
                        current_start
                    )
                    chunks.extend(sub_chunks)
                else:
                    # 直接添加块
                    chunks.append(TextChunk(
                        content=chunk_content,
                        metadata=ChunkMetadata(
                            chunk_id=f"{source_file}_adaptive_{len(chunks)}",
                            source_file=source_file,
                            word_count=len(chunk_content.split()),
                            start_index=current_start,
                            end_index=text.find(chunk_content) + len(chunk_content),
                            strategy=ChunkingStrategy.BY_ADAPTIVE,
                            content_type=current_type,
                            quality_score=self._calculate_quality_score(chunk_content, current_type)
                        )
                    ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"自适应分块失败: {str(e)}")
            raise
    
    def _analyze_content_types(self, lines: List[str]) -> Dict[str, int]:
        """
        分析文本内容类型分布
        
        Args:
            lines: 文本行列表
            
        Returns:
            Dict[str, int]: 内容类型统计
        """
        content_types = {}
        for line in lines:
            content_type = self._detect_content_type(line)
            content_types[content_type] = content_types.get(content_type, 0) + 1
        return content_types
    
    def _detect_content_type(self, line: str) -> str:
        """
        检测单行文本的内容类型
        
        Args:
            line: 文本行
            
        Returns:
            str: 内容类型
        """
        line = line.strip()
        if not line:
            return "empty"
        
        # 检测代码
        if re.match(r'^\s*(def|class|import|from|if|for|while|try|except|with|async|await)\s', line):
            return "code"
        
        # 检测表格
        if re.match(r'^\s*\|.*\|.*\|', line) or re.match(r'^\s*\+[-+]+\+', line):
            return "table"
        
        # 检测标题
        if re.match(r'^#+\s', line) or re.match(r'^[A-Z][^a-z]', line):
            return "title"
        
        # 检测列表
        if re.match(r'^\s*[-*+]\s', line) or re.match(r'^\s*\d+\.\s', line):
            return "list"
        
        # 默认为普通文本
        return "text"
    
    async def _sub_chunk_by_type(
        self,
        content: str,
        content_type: str,
        max_size: int,
        source_file: str,
        start_index: int
    ) -> List[TextChunk]:
        """
        根据内容类型进行子分块
        
        Args:
            content: 要分块的内容
            content_type: 内容类型
            max_size: 最大分块大小
            source_file: 源文件名
            start_index: 起始索引
            
        Returns:
            List[TextChunk]: 分块结果
        """
        if content_type == "code":
            # 代码按函数/类分块
            return await self._chunk_code(content, max_size, source_file, start_index)
        elif content_type == "table":
            # 表格按行分块
            return await self._chunk_table(content, max_size, source_file, start_index)
        else:
            # 其他类型使用段落分块
            return await self._chunk_by_paragraph(
                content,
                max_size,
                self.config.overlap,
                source_file
            )
    
    async def _chunk_code(
        self,
        content: str,
        max_size: int,
        source_file: str,
        start_index: int
    ) -> List[TextChunk]:
        """代码分块"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in content.split('\n'):
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_size and current_chunk:
                # 保存当前块
                chunk_content = '\n'.join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_code_{len(chunks)}",
                        source_file=source_file,
                        word_count=len(chunk_content.split()),
                        start_index=start_index,
                        end_index=start_index + len(chunk_content),
                        strategy=ChunkingStrategy.BY_ADAPTIVE,
                        content_type="code",
                        quality_score=1.0
                    )
                ))
                start_index += len(chunk_content) + 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
        
        # 处理最后一个块
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    chunk_id=f"{source_file}_code_{len(chunks)}",
                    source_file=source_file,
                    word_count=len(chunk_content.split()),
                    start_index=start_index,
                    end_index=start_index + len(chunk_content),
                    strategy=ChunkingStrategy.BY_ADAPTIVE,
                    content_type="code",
                    quality_score=1.0
                )
            ))
        
        return chunks
    
    async def _chunk_table(
        self,
        content: str,
        max_size: int,
        source_file: str,
        start_index: int
    ) -> List[TextChunk]:
        """表格分块"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_size and current_chunk:
                # 保存当前块
                chunk_content = '\n'.join(current_chunk)
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        chunk_id=f"{source_file}_table_{len(chunks)}",
                        source_file=source_file,
                        word_count=len(chunk_content.split()),
                        start_index=start_index,
                        end_index=start_index + len(chunk_content),
                        strategy=ChunkingStrategy.BY_ADAPTIVE,
                        content_type="table",
                        quality_score=1.0
                    )
                ))
                start_index += len(chunk_content) + 1
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
        
        # 处理最后一个块
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(TextChunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    chunk_id=f"{source_file}_table_{len(chunks)}",
                    source_file=source_file,
                    word_count=len(chunk_content.split()),
                    start_index=start_index,
                    end_index=start_index + len(chunk_content),
                    strategy=ChunkingStrategy.BY_ADAPTIVE,
                    content_type="table",
                    quality_score=1.0
                )
            ))
        
        return chunks
    
    def _calculate_quality_score(self, content: str, content_type: str) -> float:
        """
        计算分块质量得分
        
        Args:
            content: 分块内容
            content_type: 内容类型
            
        Returns:
            float: 质量得分
        """
        # 基础得分
        score = 1.0
        
        # 根据内容类型调整得分
        if content_type == "code":
            # 检查代码完整性
            if not re.search(r'def|class', content):
                score *= 0.8
        elif content_type == "table":
            # 检查表格完整性
            if not re.search(r'\|.*\|', content):
                score *= 0.8
        elif content_type == "text":
            # 检查文本质量
            words = content.split()
            if len(words) < 10:
                score *= 0.9
            if len(words) > 1000:
                score *= 0.9
        
        return score