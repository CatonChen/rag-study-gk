# RAG 智能问答系统

基于 RAG（检索增强生成）框架的智能问答系统，支持文档处理、语义搜索和智能问答功能。

## 功能特点

- 文档处理：支持 PDF、TXT、DOCX 等格式文档的上传和处理
- 文档分块：支持多种分块策略，保留上下文信息
- 向量化：支持文档向量化和向量存储
- 语义搜索：基于向量相似度的语义搜索
- 智能问答：基于检索结果的智能问答生成
- 流式输出：支持流式生成回答
- 缓存机制：支持结果缓存和自动清理
- 异步处理：支持并发请求处理
- 持久化存储：支持向量数据和缓存数据的持久化

## 技术栈

### 后端
- FastAPI：高性能异步 Web 框架
- Python 3.10.12：核心编程语言
- LangChain：LLM 应用开发框架
- ChromaDB：向量数据库
- Sentence Transformers：文本向量化
- OpenAI API：大语言模型接口
- Redis：缓存服务
- Celery：异步任务队列
- SQLAlchemy：ORM 框架
- Alembic：数据库迁移工具
- Pytest：单元测试框架

### 前端
- React + Vite：前端框架和构建工具
- TypeScript：类型安全的 JavaScript 超集
- Tailwind CSS：原子化 CSS 框架
- Axios：HTTP 客户端
- ESLint：代码规范检查
- Prettier：代码格式化

## 项目结构

```
project/
├── backend/
│   ├── 01-loaded-docs/    # 原始文档存储
│   ├── 01-chunked-docs/   # 分块文档存储
│   ├── 02-embedded-docs/  # 向量化文档存储
│   ├── 03-vector-store/   # 向量数据库存储
│   ├── 04-search-results/ # 搜索结果存储
│   ├── 05-generation-results/ # 生成结果存储
│   ├── services/          # 核心服务实现
│   │   ├── embedding_service.py    # 向量化服务
│   │   ├── vector_store_service.py # 向量存储服务
│   │   ├── search_service.py       # 搜索服务
│   │   ├── cache_service.py        # 缓存服务
│   │   └── generation_service.py   # 生成服务
│   ├── tests/            # 单元测试
│   ├── utils/            # 工具函数
│   └── main.py           # 应用入口
└── frontend/
    ├── src/
    │   ├── components/   # 可复用组件
    │   ├── pages/       # 页面组件
    │   ├── config/      # 配置文件
    │   └── assets/      # 静态资源
    ├── package.json     # 项目配置
    └── vite.config.ts   # Vite 配置
```

## 快速开始

### 环境要求
- Python 3.10.12
- Node.js 16+
- Redis 6.0+
- PostgreSQL 13+
- OpenAI API Key

### 安装步骤

1. 克隆项目
```bash
git clone [项目地址]
cd rag-study-gk
```

2. 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

3. 安装前端依赖
```bash
cd frontend
npm install
```

4. 配置环境变量
```bash
# 复制环境变量示例文件
cp backend/.env.example backend/.env
# 编辑 .env 文件，填入必要的配置信息
```

5. 启动服务
```bash
# 启动后端服务
cd backend
uvicorn main:app --reload

# 启动前端服务
cd frontend
npm run dev
```

## 核心服务

### 1. 向量化服务 (EmbeddingService)
- 支持多种嵌入模型
- 批量文本向量化
- 异步处理支持
- 错误处理和重试机制

### 2. 向量存储服务 (VectorStoreService)
- 集合管理（创建、删除、列表）
- 向量操作（添加、搜索、更新、删除）
- 持久化存储
- 元数据管理

### 3. 搜索服务 (SearchService)
- 语义搜索
- 相似度阈值过滤
- 结果重排序
- 搜索历史记录

### 4. 缓存服务 (CacheService)
- 内存缓存
- 持久化存储
- 自动过期清理
- 并发访问支持

## API 文档

启动后端服务后，访问 http://localhost:8000/docs 查看完整的 API 文档。

### 主要接口

#### 文档处理
- POST /process - 处理上传文档
- POST /save - 保存处理结果
- GET /list-docs - 获取文档列表

#### 向量化
- POST /embed - 文档向量化
- GET /list-embedded - 获取向量化文档列表

#### 检索
- POST /search - 执行语义搜索
- GET /collections - 获取向量集合列表
- GET /collections/{provider} - 获取特定提供商的集合

#### 生成
- POST /generate - 生成回答
- GET /generation/models - 获取可用模型列表

## 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_embedding_service.py

# 运行带覆盖率报告的测试
pytest --cov=services tests/
```

### 测试覆盖
- 单元测试：核心服务功能测试
- 集成测试：服务间交互测试
- 性能测试：并发和负载测试

## 部署

### Docker 部署
```bash
# 构建镜像
docker build -t rag-system .

# 运行容器
docker run -d -p 8000:8000 -p 3000:3000 rag-system
```

### 环境变量配置
- OPENAI_API_KEY：OpenAI API 密钥
- HOST：服务器主机地址
- PORT：服务器端口
- DEBUG：调试模式开关
- REDIS_URL：Redis 连接地址
- DATABASE_URL：数据库连接地址
- VECTOR_STORE_DIR：向量存储目录
- CACHE_DIR：缓存存储目录

## 开发指南

### 后端开发
- 遵循 PEP 8 编码规范
- 使用类型注解
- 编写单元测试
- 使用异步编程
- 实现错误处理
- 添加日志记录

### 前端开发
- 使用 TypeScript
- 遵循 ESLint 规范
- 使用函数组件和 Hooks
- 实现响应式设计
- 添加错误边界
- 优化性能

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License 