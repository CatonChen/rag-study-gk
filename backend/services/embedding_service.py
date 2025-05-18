from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from tqdm import tqdm

class EmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_dir: str = "02-embedded-docs",
        collection_name: str = "documents"
    ):
        self.model = SentenceTransformer(model_name)
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # 初始化 ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=str(self.vector_dir / "chroma"),
            anonymized_telemetry=False
        ))
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """将文本转换为向量"""
        return self.model.encode(text)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文本块转换为向量"""
        embedded_chunks = []
        
        for chunk in tqdm(chunks, desc="Embedding chunks"):
            vector = self.embed_text(chunk["text"])
            embedded_chunk = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "vector": vector.tolist(),
                "metadata": chunk["metadata"]
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def save_vectors(self, vectors: List[Dict[str, Any]], output_file: str):
        """保存向量到文件"""
        output_path = self.vector_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vectors, f, ensure_ascii=False, indent=2)
    
    def load_vectors(self, file_path: str) -> List[Dict[str, Any]]:
        """从文件加载向量"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def add_to_collection(self, chunks: List[Dict[str, Any]]):
        """将向量添加到 ChromaDB 集合"""
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [
            {
                "chunk_id": id,
                "text": doc,
                "metadata": metadata,
                "distance": distance
            }
            for id, doc, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        return {
            "collection_name": self.collection_name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }
    
    def delete_collection(self):
        """删除集合"""
        self.client.delete_collection(self.collection_name) 