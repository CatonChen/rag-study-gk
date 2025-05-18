# backend/services 目录说明

本目录包含 RAG 智能问答系统后端的所有核心服务模块。每个服务均为独立的 Python 模块，负责系统的不同核心功能。下文将详细介绍每个服务的作用及主要接口。

---

## 1. 文档加载服务（loading_service.py）
- **功能**：负责加载本地或远程的 PDF、TXT、DOCX 等格式文档，支持批量文件上传。
- **主要特性**：
  - 支持多种文档格式
  - 提取文件元数据（文件名、类型、大小、页数等）
  - 异常处理与日志记录
- **典型接口**：
  - `load_document(path: str) -> Document`
  - `batch_load(paths: List[str]) -> List[Document]`

---

## 2. 文档分块服务（chunking_service.py）
- **功能**：将文档内容按不同策略进行分块，便于后续向量化和检索。
- **主要特性**：
  - 支持按段落、句子、字符数等多种分块方式
  - 支持分块重叠，保留上下文
  - 分块结构化输出，包含分块元信息
- **典型接口**：
  - `chunk_by_paragraph(doc: Document) -> List[Chunk]`
  - `chunk_by_sentence(doc: Document) -> List[Chunk]`
  - `chunk_by_length(doc: Document, length: int) -> List[Chunk]`

---

## 3. 文档解析服务（parsing_service.py）
- **功能**：对文档内容进行深度解析，支持多种解析策略。
- **主要特性**：
  - 全文提取、逐页解析、按标题分段、文本与表格混合解析
  - 支持文档元数据提取、页面映射、页码追踪
  - 错误处理与详细日志
  - 结构化内容输出
- **典型接口**：
  - `parse_document(path: str, strategy: str) -> List[ParsedContent]`
  - `process_batch(paths: List[str], strategy: str) -> Dict[str, Result]`

---

## 4. 向量嵌入服务（embedding_service.py）
- **功能**：将文本内容转换为向量表示，支持多种嵌入模型。
- **主要特性**：
  - 支持 OpenAI、HuggingFace 等模型
  - 支持批量向量化
  - 可配置向量维度
  - 异步处理
- **典型接口**：
  - `get_embedding(text: str) -> List[float]`
  - `batch_embed(texts: List[str]) -> List[List[float]]`

---

## 5. 向量存储服务（vector_store_service.py）
- **功能**：负责向量的存储、检索、更新和删除，底层支持 ChromaDB。
- **主要特性**：
  - 支持集合管理（创建、删除、列表）
  - 支持向量的增删改查
  - 支持元数据管理和持久化
- **典型接口**：
  - `create_collection(name: str) -> str`
  - `add_vectors(collection: str, vectors: List[List[float]], docs: List[str], metadatas: List[dict])`
  - `search(collection: str, query_vector: List[float], top_k: int) -> List[dict]`
  - `update_vectors(collection: str, ids: List[str], vectors: List[List[float]])`
  - `delete_vectors(collection: str, ids: List[str])`

---

## 6. 检索服务（search_service.py）
- **功能**：基于向量数据库实现语义检索，支持相似度阈值、结果排序和过滤。
- **主要特性**：
  - 支持语义搜索与重排序
  - 支持搜索历史与统计
  - 支持结果缓存
- **典型接口**：
  - `search(query: str, collection: str, filters: dict) -> List[SearchResult]`
  - `get_search_history(collection: str) -> List[dict]`
  - `clear_cache(collection: str)`

---

## 7. 生成服务（generation_service.py）
- **功能**：调用大语言模型（如 OpenAI GPT）基于检索结果生成最终回答。
- **主要特性**：
  - 支持多种 LLM
  - 支持上下文管理与参数配置
  - 支持流式输出
- **典型接口**：
  - `generate_answer(query: str, context: List[str], model: str) -> str`

---

## 8. 缓存服务（cache_service.py）
- **功能**：为检索、生成等高频操作提供缓存能力，提升系统性能。
- **主要特性**：
  - 支持内存和磁盘缓存
  - 支持 TTL 过期与自动清理
  - 支持并发访问
- **典型接口**：
  - `get(key: str) -> Any`
  - `set(key: str, value: Any, ttl: int)`
  - `delete(key: str)`
  - `clear()`

---

## 9. 元数据与监控服务（metadata_service.py, monitoring_service.py）
- **功能**：
  - 元数据服务：统一管理文档、分块、向量等元信息
  - 监控服务：记录系统运行状态、性能指标、错误日志等
- **主要特性**：
  - 支持元数据的增删查改
  - 支持系统健康检查与性能统计

---

## 目录结构建议
- 每个服务文件均有详细注释和类型标注
- 推荐为每个服务编写单元测试（见 ../tests/）
- 如需扩展新服务，建议遵循本目录结构和接口风格

---

如需详细接口文档，请参考每个服务文件内的 docstring 说明。 