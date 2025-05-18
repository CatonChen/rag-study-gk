# RAG 项目框架构建提示词

## 项目概述
请帮我构建一个基于 RAG（检索增强生成）框架的智能问答系统，该系统需要能够处理文档、进行语义搜索并生成回答。项目采用前后端分离架构。

## 技术栈要求

### 后端
- 使用 FastAPI 框架
- Python 3.10.12
- 需要支持异步操作
- 使用 Pydantic 进行数据验证
- 实现 CORS 支持

### 前端
- 使用 React + Vite
- 使用 Tailwind CSS 进行样式管理
- 使用 ESLint 进行代码规范
- 支持响应式设计

## 核心功能模块

### 1. 文档处理模块
- 文档加载服务（LoadingService）
  - 支持 PDF、TXT、DOCX 等格式
  - 支持批量文件上传
  - 提供文件元数据提取

- 文档分块服务（ChunkingService）
  - 支持多种分块策略（按段落、按句子、按字符数）
  - 支持重叠分块
  - 保留分块上下文信息

- 文档解析服务（ParsingService）
  - 支持多种解析策略：
    - 全文提取（all_text）
    - 逐页解析（by_pages）
    - 基于标题的分段（by_titles）
    - 文本和表格混合解析（text_and_tables）
  - 支持文档元数据提取
  - 支持页面映射和页码追踪
  - 提供错误处理和日志记录
  - 支持结构化内容输出

### 2. 向量化模块
- 向量嵌入服务（EmbeddingService）
  - 支持多种嵌入模型（如 OpenAI、HuggingFace）
  - 支持批量向量化
  - 提供向量维度配置

- 向量存储服务（VectorStoreService）
  - 支持多种向量数据库（如 Chroma、Milvus）
  - 提供索引管理
  - 支持向量更新和删除

### 3. 检索模块
- 搜索服务（SearchService）
  - 支持语义搜索
  - 支持相似度阈值设置
  - 支持结果排序和过滤
  - 提供搜索结果评估

### 4. 生成模块
- 生成服务（GenerationService）
  - 支持多种 LLM 模型
  - 支持上下文管理
  - 提供生成参数配置
  - 支持流式输出

## 数据存储结构
```
project/
├── backend/
│   ├── 01-loaded-docs/    # 原始文档
│   ├── 01-chunked-docs/   # 分块文档
│   ├── 02-embedded-docs/  # 向量化文档
│   ├── 03-vector-store/   # 向量存储
│   ├── 04-search-results/ # 搜索结果
│   ├── 05-generation-results/ # 生成结果
│   ├── services/          # 核心服务
│   └── utils/            # 工具函数
└── frontend/
    ├── src/
    │   ├── components/   # 可复用组件
    │   ├── pages/       # 页面组件
    │   ├── config/      # 配置文件
    │   └── assets/      # 静态资源
```

## API 接口要求

### 文档处理接口
- POST /process - 处理上传文档
- POST /save - 保存处理结果
- GET /list-docs - 获取文档列表

### 向量化接口
- POST /embed - 文档向量化
- GET /list-embedded - 获取向量化文档列表

### 检索接口
- POST /search - 执行语义搜索
- GET /collections - 获取向量集合列表
- GET /collections/{provider} - 获取特定提供商的集合

### 生成接口
- POST /generate - 生成回答
- GET /generation/models - 获取可用模型列表

## 前端界面要求

### 1. 文档管理页面
- 文件上传区域
- 文档列表展示
- 处理状态显示
- 文档预览功能

### 2. 检索页面
- 搜索输入框
- 搜索结果展示
- 相似度显示
- 结果筛选选项

### 3. 生成页面
- 问题输入区域
- 上下文显示
- 生成结果展示
- 模型选择选项

## 安全性要求
- 实现 API 密钥管理
- 配置 CORS 策略
- 实现错误处理机制
- 添加请求速率限制

## 性能要求
- 支持异步处理
- 实现缓存机制
- 优化向量搜索性能
- 支持批量处理

## 部署要求
- 提供 Docker 配置
- 提供环境变量配置
- 提供部署文档
- 支持容器化部署

## 文档要求
- API 文档（使用 Swagger/OpenAPI）
- 部署文档
- 使用说明文档
- 开发文档

请根据以上要求，帮我构建一个完整的 RAG 项目框架。项目应该具有良好的可扩展性、可维护性和性能表现。 