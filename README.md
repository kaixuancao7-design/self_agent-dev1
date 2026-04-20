# AI Agent 智能知识库系统

一个基于 LangChain 框架构建的智能知识库 AI Agent 系统，具备从被动检索到主动规划的高级能力。

## ✨ 核心功能

### 📚 知识库管理
- 支持多种文件格式上传（PDF、DOCX、TXT、MD、XLSX、CSV、HTML、图片等）
- 多数据库管理（创建、选择、删除知识库）
- 文件列表展示、删除、重新解析
- 进度条显示上传状态
- 拖拽上传支持

### 🔍 智能检索
- 基于 Chroma 向量数据库的语义检索
- BM25 关键词检索
- 混合检索策略（Hybrid Retrieval）
- **重排功能（Reranking）**：支持 Linear、Cross-Encoder、LLM 三种重排策略
- **Firecrawl MCP**：网页内容抓取，支持 JavaScript 渲染页面
- 图片内容索引与检索

### 🤖 高级 Agent 能力

| 能力 | 描述 |
|------|------|
| **任务拆解** | 将复杂目标自动拆解为可执行的子任务序列 |
| **路径规划** | 为子任务排序，确定执行逻辑和优先级 |
| **自我评估** | 评估每步结果质量，自动调整策略 |
| **事实核查** | 校验答案与参考资料一致性，避免幻觉 |
| **动态调整** | 处理失败后的备选方案 |

### 🧠 记忆管理系统

| 类型 | 描述 |
|------|------|
| **短期记忆** | 跟踪当前会话上下文，支持流畅的多轮对话 |
| **长期记忆** | 通过 RAG 访问外部知识库，记住用户偏好、历史任务进度和关键业务信息 |

### 🔍 意图识别能力

| 功能 | 描述 |
|------|------|
| **显式意图识别** | 通过规则匹配识别常见意图（知识查询、投诉分析、任务请求等） |
| **隐含意图分析** | 深度理解用户真实需求，不仅仅是关键词匹配 |
| **智能响应** | 根据识别到的意图，自动选择合适的工具和策略 |

**支持的意图类型:**
- `knowledge_query`: 知识查询（如"什么是AI Agent？"）
- `complaint_analysis`: 投诉分析（如"最近客户投诉变多了"）
- `task_request`: 任务请求（如"帮我制定方案"）
- `data_retrieval`: 数据检索（如"查询历史记录"）
- `conversation`: 闲聊对话（如"你好"）
- `preference_setting`: 偏好设置（如"设置我的偏好"）

### 🔧 Skills 框架（Anthropic 标准）

系统基于 **Anthropic Skills** 标准框架构建，每个技能包含：
- **SKILL.md**: 技能说明文档（能力描述、参数、返回值、使用场景）
- **reference/**: 详细参考文档（API说明、实现细节）
- **script/**: 可执行脚本（技能实现代码）

### 技能分类

| 分类 | 技能 | 功能 | 位置 |
|------|------|------|------|
| **核心检索** | rag_summarize | 知识库语义检索 | skills/ |
| **网络搜索** | web_search | 实时网络搜索 | skills/ |
| **通用工具** | get_weather | 天气查询 | skills/ |
| **高级规划** | task_decompose | 任务拆解 | skills/ |
| **高级规划** | fact_check | 事实核查 | skills/ |
| **知识库管理** | kb_management | 数据库管理 | skills/ |
| **智能内容生成** | content_generator | PPT/思维导图/报告生成 | skills/ |
| **文档处理** | document_processor | PDF/Word/Excel解析 | skills/ |
| **知识治理** | knowledge_governance | 质量报告/统计分析 | skills/ |
| **报告生成** | get_user_history | 用户历史 | tools/ |
| **报告生成** | fill_context_report | 报告切换 | tools/ |
| **通用工具** | get_user_location | 用户定位 | tools/ |
| **通用工具** | get_user_id | 用户ID | tools/ |
| **通用工具** | get_current_month | 当前月份 | tools/ |
| **高级规划** | evaluate_result | 结果评估 | tools/ |
| **网络搜索** | fetch_webpage | 获取网页 | tools/ |
| **网络搜索** | firecrawl_scrape | Firecrawl网页抓取 | skills/ |
| **网络搜索** | firecrawl_crawl | Firecrawl网站爬取 | skills/ |
| **网络搜索** | firecrawl_search | Firecrawl搜索 | skills/ |
| **知识库管理** | 细粒度管理 | 数据库操作 | tools/ |

### 🔗 统一接口层（Unified API Abstraction）

系统提供统一的LLM和Embedding调用接口，屏蔽不同Provider的实现差异。

**核心抽象:**

| 抽象类 | 描述 | 核心方法 |
|--------|------|----------|
| `BaseLLM` | LLM抽象基类 | `chat(messages)`, `generate(prompt)`, `chat_stream(messages)` |
| `BaseEmbedding` | Embedding抽象基类 | `embed(texts)`, `embed_single(text)` |
| `BaseVisionLLM` | Vision LLM抽象基类 | `chat_with_image(messages, image_urls)` |

**Provider支持:**

| Provider | 典型场景 | 配置切换点 |
|----------|----------|------------|
| **Azure OpenAI** | 企业合规、私有云部署、区域数据驻留 | `provider: azure` |
| **OpenAI 原生** | 通用开发、最新模型尝鲜 | `provider: openai` |
| **DeepSeek** | 成本优化、特定语言优化 | `provider: deepseek` |
| **Ollama/vLLM** | 完全离线、隐私敏感、无API成本 | `provider: ollama` |

**推荐模型配置:**
- **LLM**: `qwen3.5:9b`（Ollama）
- **Embedding**: `Mxbai-embed-large`（Ollama，支持1024维度向量）

**工厂模式:**
- `LLMFactory.create(provider, **kwargs)` - 创建LLM实例
- `EmbeddingFactory.create(provider, **kwargs)` - 创建Embedding实例
- `VisionLLMFactory.create(provider, **kwargs)` - 创建Vision LLM实例

### 🔄 重排策略（Reranking）

系统支持多种重排策略，提升检索结果的相关性：

| 重排方法 | 实现方式 | 特点 | 适用场景 |
|----------|----------|------|----------|
| **Linear** | 规则线性融合 | 轻量、快速、无需额外模型 | 快速原型、低延迟场景 |
| **Cross-Encoder** | BAAI/bge-reranker-base | 专业重排模型，效果好 | 生产环境、追求效果 |
| **LLM** | 大语言模型精排 | 理解深层语义，成本较高 | 需要深层语义理解 |

**重排配置:**
```yaml
# config/rag.yml
rerank:
  method: linear  # linear/cross_encoder/llm
  cross_encoder_model: BAAI/bge-reranker-base
```

**检索流程:**
```
Query → Query Processing → Hybrid Search → RRF Fusion → Filtering → Reranking → Top-K Results
```

### 📊 Ragas 量化评估

系统集成 Ragas 风格的量化评估功能，用于评估检索与生成效果，并通过 Bad Case 分析持续优化。

**评估指标:**

| 类别 | 指标 | 说明 |
|------|------|------|
| **检索效果** | Hit Rate | 是否检索到至少一条相关文档 |
| | MRR | 平均倒数排名 |
| | Recall@k | 召回率 |
| | Precision@k | 精确率 |
| | Diversity | 文档多样性 |
| **生成效果** | Faithfulness | 事实一致性（答案与文档的一致性） |
| | Answer Relevance | 答案相关性（答案与问题的相关性） |
| | Context Utilization | 上下文利用率 |
| | Answer Correctness | 答案正确性 |
| | Answer Conciseness | 答案简洁性 |

**Bad Case 分析:**
- 自动识别问题类型：检索失败、低相关性、幻觉、答案不相关、生成过度/不足
- 提供针对性优化建议
- 支持分块策略、提示工程、召回逻辑的优化指导

**评估命令:**
```bash
python scripts/run_ragas_evaluation.py
```

### 🌐 FastAPI 后端接口

系统提供完整的 RESTful API 接口，支持知识库CRUD、Agent问答、文件上传等核心功能。

**主要API端点:**

| 模块 | 端点 | 方法 | 描述 |
|------|------|------|------|
| **健康检查** | `/api/health` | GET | 服务健康检查 |
| **用户管理** | `/api/users` | POST | 创建用户 |
| | `/api/users/{user_id}` | GET | 获取用户信息 |
| | `/api/users/by-username/{username}` | GET | 根据用户名获取用户 |
| **会话管理** | `/api/sessions` | POST | 创建会话 |
| | `/api/sessions/{session_id}` | GET | 获取会话信息 |
| | `/api/sessions/{session_id}` | DELETE | 删除会话 |
| **知识库** | `/api/databases` | GET | 获取数据库列表 |
| | `/api/databases/{db_name}` | POST | 创建数据库 |
| | `/api/databases/{db_name}` | DELETE | 删除数据库 |
| | `/api/databases/current` | GET | 获取当前数据库信息 |
| **文件上传** | `/api/upload` | POST | 上传文件到知识库 |
| | `/api/upload/celery` | POST | 通过Celery异步上传 |
| **聊天** | `/api/chat` | POST | Agent问答接口 |

### 🔐 多用户会话隔离

系统支持多用户会话隔离，确保数据安全性：

| 特性 | 描述 |
|------|------|
| **用户表** | 存储用户信息（ID、用户名、邮箱） |
| **会话表** | 存储会话信息（ID、用户ID、会话名称、创建时间） |
| **数据隔离** | 每个用户的会话独立，数据不串流 |
| **状态持久化** | 支持本地文件和数据库两种持久化方式 |

### ⚡ 异步架构

系统采用异步设计，提升并发处理能力：

| 特性 | 描述 |
|------|------|
| **异步数据库操作** | 使用 asyncpg/sqlalchemy async 进行异步数据库操作 |
| **异步LLM调用** | 支持异步调用LLM接口 |
| **Celery任务队列** | 处理长任务（如大文件解析、Ragas评估） |
| **并发支持** | 支持10人以内并发，无明显卡顿 |

## 🏗️ 项目结构

```
├── agent/                    # Agent 核心模块
│   ├── modules/             # 高级能力模块
│   ├── skills/              # Skills 框架（Anthropic 标准）
│   ├── tools/               # 传统工具模块
│   ├── react_agent.py       # 主 Agent 实现
│   ├── langgraph_workflow.py # LangGraph工作流引擎
│   └── session.py           # 会话管理
├── model/                   # 统一接口层
│   ├── base.py              # 抽象基类
│   ├── factory.py           # 工厂类
│   └── providers/           # Provider实现
├── rag/                     # RAG 检索服务
├── backend/                 # FastAPI 后端服务
├── scripts/                 # 运行脚本
├── config/                  # 配置文件
├── prompts/                 # 提示词文件
├── chromadb/                # Chroma 向量数据库
├── data/                    # 数据文件
├── utils/                   # 工具函数
├── test/                    # 测试文件
└── app.py                   # Streamlit 应用入口
```

## 🏗️ 整体架构设计

### 架构总览

系统采用 **模块化、分层架构** 设计，基于 LangChain + LangGraph 构建，支持多Provider切换、混合检索、多用户会话隔离等核心能力。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         应用层 (Application Layer)                      │
│  ┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐  │
│  │ Streamlit    │    │      FastAPI API     │    │   Celery Tasks   │  │
│  │   前端界面   │    │      后端接口层       │    │    任务队列      │  │
│  └──────┬───────┘    └──────────┬───────────┘    └────────┬─────────┘  │
└─────────┼────────────────────────┼─────────────────────────┼───────────┘
          │                        │                         │
┌─────────▼────────────────────────▼─────────────────────────▼───────────┐
│                        业务逻辑层 (Business Logic Layer)                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      ReactAgent / LangGraph                      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │  │
│  │  │ 意图识别 │→│任务拆解  │→│路径规划  │→│ 执行/评估│→│ 事实核查  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  │                              │                                     │
│  │                    ┌─────────▼─────────┐                           │
│  │                    │   Skills 框架     │                           │
│  │                    │ (自动加载/降级)   │                           │
│  │                    └───────────────────┘                           │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                          │
┌──────────────────────────────────────────▼──────────────────────────────┐
│                       数据层 (Data Layer)                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   Chroma     │    │   SQLite     │    │   Cache      │             │
│  │  向量数据库   │    │  用户/会话表  │    │   缓存层     │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
                                          │
┌──────────────────────────────────────────▼──────────────────────────────┐
│                      模型层 (Model Layer)                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Unified API Abstraction (统一接口层)                 │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │  │
│  │  │  Ollama  │ │ OpenAI   │ │  Azure   │ │ DeepSeek │           │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 模块职责划分

| 层级 | 模块 | 职责描述 |
|------|------|----------|
| **应用层** | Streamlit | 可视化前端界面，提供用户交互 |
| | FastAPI | RESTful API 接口，支持外部调用 |
| | Celery | 异步任务队列，处理长任务 |
| **业务逻辑层** | ReactAgent | 主Agent实现，集成工具调用 |
| | LangGraph | 工作流引擎，管理执行流程 |
| | Skills 框架 | 技能自动加载与失败降级 |
| | 高级能力模块 | 任务拆解、路径规划、自我评估等 |
| | 记忆管理 | 短期记忆+长期记忆 |
| **数据层** | Chroma | 向量数据库，存储文档嵌入 |
| | SQLite | 用户表、会话表 |
| | Cache | 检索结果缓存 |
| **模型层** | Unified API | 统一LLM/Embedding调用接口 |

### 核心数据流

#### 问答流程

```
用户输入 → 意图识别 → [简单问题] → RAG检索 → 直接回答
                         ↓
                    [复杂问题] → 任务拆解 → 路径规划 → 分步执行 → 结果整合 → 事实核查 → 回答
```

#### 检索流程

```
Query → Query Processing → Hybrid Search → RRF Fusion → Filtering → Reranking → Top-K Results
         ↓                    ↓
    同义词扩展          BM25 + 向量检索
```

#### 文件上传流程

```
文件上传 → 临时存储 → 文档解析 → 分块处理 → 空Chunk过滤 → 向量嵌入 → 存储到Chroma
                                                   ↓
                                           记录到SQLite
```

### 关键设计模式

#### 工厂模式（Factory Pattern）

用于创建不同Provider的模型实例：

```python
# model/factory.py
class LLMFactory:
    @staticmethod
    def create(provider, **kwargs):
        if provider == "ollama":
            return OllamaProvider(**kwargs)
        elif provider == "openai":
            return OpenAIProvider(**kwargs)
```

**优点**：屏蔽底层实现差异，支持运行时切换Provider，符合开闭原则。

#### 策略模式（Strategy Pattern）

用于重排策略的切换：

```python
# rag/reranker.py
class RerankerFactory:
    @staticmethod
    def create(method):
        if method == "linear":
            return LinearReranker()
        elif method == "cross_encoder":
            return CrossEncoderReranker()
        elif method == "llm":
            return LLMReranker()
```

#### 代理模式（Proxy Pattern）

用于缓存和重试逻辑：

```python
# 缓存代理
def retrieve_docs(query):
    if cache_enabled:
        cached = cache_service.get(query)
        if cached:
            return cached
    result = actual_retrieve(query)
    cache_service.set(query, result)
    return result
```

#### 状态模式（State Pattern）

LangGraph工作流中使用状态机管理执行流程：

```python
# agent/langgraph_workflow.py
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    step_count: int = 0
    retry_count: int = 0
    is_finished: bool = False
```

### 技术栈

| 分类 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **框架** | LangChain | 0.1.x | Agent框架 |
| | LangGraph | 0.1.x | 工作流引擎 |
| | FastAPI | 0.100+ | 后端API |
| | Streamlit | 1.30+ | 前端界面 |
| **数据库** | Chroma | 0.4+ | 向量数据库 |
| | SQLite | 内置 | 用户/会话存储 |
| **模型** | Ollama | 0.1.x | 本地LLM |
| | OpenAI/Azure | API | 云端LLM |
| **异步** | Celery | 5.3+ | 任务队列 |
| **工具** | Firecrawl | API | 网页抓取 |

### 关键特性

#### 多用户会话隔离

```
用户A ──→ 会话A1 ──→ 数据A
        └─→ 会话A2 ──→ 数据A
        
用户B ──→ 会话B1 ──→ 数据B
        └─→ 会话B2 ──→ 数据B
```

- 用户表 + 会话表设计
- 会话数据独立存储
- 状态持久化支持

#### 异步架构

| 特性 | 实现方式 |
|------|----------|
| 异步数据库操作 | SQLAlchemy Async |
| 异步LLM调用 | aiohttp |
| 长任务处理 | Celery + Redis |

#### 可配置性

- **模型Provider**：可配置切换（Ollama/OpenAI/Azure/DeepSeek）
- **检索权重**：BM25/向量/RRF权重可配置
- **重排策略**：Linear/Cross-Encoder/LLM可配置
- **工作流参数**：最大步数、检查点、日志级别可配置

### 安全性与可靠性

#### 安全措施

| 措施 | 说明 |
|------|------|
| **输入验证** | 所有API输入进行校验 |
| **SQL注入防护** | 使用ORM参数化查询 |
| **日志脱敏** | 敏感信息不记录 |
| **API密钥管理** | 环境变量存储 |

#### 可靠性保障

| 措施 | 说明 |
|------|------|
| **失败重试** | 模型调用失败自动重试 |
| **技能降级** | 技能失败时返回友好提示 |
| **超时处理** | 请求超时限制 |
| **最大步数限制** | 防止LangGraph死循环 |

### 扩展性设计

#### 添加新技能

1. 在 `agent/skills/` 创建技能目录
2. 创建 `SKILL.md`、`reference/`、`script/main.py`
3. SkillLoader 自动发现并加载

#### 添加新Provider

1. 在 `model/providers/` 创建Provider实现
2. 在 `model/factory.py` 注册
3. 在配置文件中添加provider选项

#### 添加新工具

1. 在 `agent/tools/` 创建工具模块
2. 在 `__init__.py` 导出
3. 在 `react_agent.py` 中注册

## 📦 主要模块详解

### 1. Agent 核心模块 (`agent/`)

Agent 模块是系统的核心，负责实现智能问答逻辑和高级能力。

#### 1.1 高级能力模块 (`agent/modules/`)

| 文件 | 功能 | 说明 |
|------|------|------|
| `task_decomposer.py` | 任务拆解 | 将复杂问题分解为多个可执行的子任务 |
| `path_planner.py` | 路径规划 | 为子任务排序，确定执行顺序和逻辑 |
| `self_evaluator.py` | 自我评估 | 评估每步执行结果的质量 |
| `fact_checker.py` | 事实核查 | 验证答案与参考文档的一致性 |
| `dynamic_adjuster.py` | 动态调整 | 处理失败后自动调整策略 |
| `memory_manager.py` | 记忆管理 | 管理短期和长期记忆 |
| `intent_recognizer.py` | 意图识别 | 识别用户意图类型 |

**工作流程:**
```
用户输入 → 意图识别 → [简单问题: 直接检索]
                   → [复杂问题: 任务拆解 → 路径规划 → 执行 → 评估 → 总结]
```

#### 1.2 Skills 框架 (`agent/skills/`)

基于 Anthropic Skills 标准构建，每个技能包含：
- **SKILL.md**: 技能说明文档
- **reference/**: 详细参考文档
- **script/main.py**: 可执行脚本

**核心技能:**
| 技能 | 功能 |
|------|------|
| `rag_summarize` | 知识库语义检索和总结 |
| `web_search` | 实时网络搜索 |
| `task_decompose` | 任务拆解 |
| `fact_check` | 事实核查 |
| `kb_management` | 知识库管理 |

**SkillLoader 自动加载机制:**
- 自动扫描 `agent/skills/` 目录
- 发现新技能并自动注册
- 支持技能失败降级（FallbackSkill）

#### 1.3 LangGraph 工作流引擎 (`agent/langgraph_workflow.py`)

基于 LangGraph 重构的 Agent 执行流程，核心特性：

| 特性 | 说明 |
|------|------|
| **状态管理** | AgentState 管理消息历史、任务列表、工具结果 |
| **最大步数限制** | 默认10步，防止死循环 |
| **异常捕获** | 捕获并处理执行过程中的异常 |
| **失败重试** | 支持失败重试机制 |
| **状态持久化** | 支持本地文件和数据库两种方式 |

**核心节点:**
```
Think → Tool Call → Evaluate → Fact Check → Summarize
```

### 2. 统一接口层 (`model/`)

提供统一的 LLM 和 Embedding 调用接口，屏蔽不同 Provider 的实现差异。

#### 2.1 抽象基类 (`model/base.py`)

| 抽象类 | 核心方法 | 说明 |
|--------|----------|------|
| `BaseLLM` | `chat()`, `generate()`, `chat_stream()` | 大语言模型抽象 |
| `BaseEmbedding` | `embed()`, `embed_single()` | 向量嵌入抽象 |
| `BaseVisionLLM` | `chat_with_image()` | 视觉模型抽象 |

#### 2.2 工厂类 (`model/factory.py`)

采用工厂模式创建模型实例：

```python
# 创建LLM实例
llm = LLMFactory.create("ollama", model="qwen3.5:9b")

# 创建Embedding实例
embedding = EmbeddingFactory.create("ollama", model="Mxbai-embed-large")
```

#### 2.3 Provider 实现 (`model/providers/`)

| Provider | 文件 | 说明 |
|----------|------|------|
| Azure OpenAI | `azure_provider.py` | 企业级部署 |
| OpenAI | `openai_provider.py` | 原生API |
| DeepSeek | `deepseek_provider.py` | 低成本方案 |
| Ollama | `ollama_provider.py` | 本地部署 |

### 3. RAG 检索服务 (`rag/`)

实现混合检索策略，提升知识库检索效果。

#### 3.1 核心组件

| 文件 | 功能 | 说明 |
|------|------|------|
| `vector_store.py` | 向量存储 | Chroma向量数据库封装，含空Chunk过滤、大文件分块、缓存机制 |
| `bm25_index.py` | BM25索引 | 关键词检索 |
| `hybrid_retriever.py` | 混合检索 | BM25+向量融合，支持RRF算法 |
| `reranker.py` | 重排器 | Linear/Cross-Encoder/LLM三种策略 |
| `ragas_evaluator.py` | 量化评估 | Ragas风格评估指标 |

#### 3.2 检索流程

```
Query → Query Processing → Hybrid Search → RRF Fusion → Filtering → Reranking → Top-K Results
```

**混合检索权重配置:**
```yaml
retrieval:
  bm25_weight: 0.5      # BM25检索权重
  vector_weight: 0.5    # 向量检索权重
  hybrid_weight: 0.5    # 混合权重
  rrf_k: 60             # RRF融合参数
```

#### 3.3 优化特性

| 特性 | 说明 |
|------|------|
| **空Chunk过滤** | 过滤长度<10的分块，提升检索质量 |
| **大文件分块** | 根据文件大小动态调整chunk_size |
| **Chroma缓存** | 提高检索性能 |
| **权重可配置** | BM25/向量/RRF权重均可调整 |

### 4. FastAPI 后端服务 (`backend/`)

提供 RESTful API 接口，支持多用户会话隔离。

#### 4.1 项目结构

| 文件 | 功能 | 说明 |
|------|------|------|
| `main.py` | 应用入口 | FastAPI配置和路由定义 |
| `database.py` | 数据库模型 | 用户表、会话表定义 |
| `async_providers.py` | 异步封装 | 异步LLM和Embedding调用 |
| `tasks.py` | Celery任务 | 处理长任务（大文件解析等） |

#### 4.2 数据库模型

**用户表 (users):**
| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| username | VARCHAR | 用户名（唯一） |
| email | VARCHAR | 邮箱 |
| created_at | DATETIME | 创建时间 |

**会话表 (sessions):**
| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| user_id | UUID | 用户ID（外键） |
| name | VARCHAR | 会话名称 |
| created_at | DATETIME | 创建时间 |

#### 4.3 API 示例

**创建用户:**
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com"}'
```

**创建会话:**
```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-uuid", "name": "我的会话"}'
```

**聊天:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-uuid", "message": "你好"}'
```

### 5. 配置系统 (`config/`)

集中管理系统配置，支持 YAML 格式。

| 文件 | 说明 |
|------|------|
| `agent.yml` | Agent配置（模型、高级能力开关） |
| `chroma.yml` | Chroma向量数据库配置 |
| `rag.yml` | RAG检索配置（权重、重排策略） |
| `prompts.yml` | 提示词配置 |

**配置加载流程:**
```
启动 → 加载config/*.yml → 解析配置 → 初始化组件
```

### 6. 提示词系统 (`prompts/`)

管理所有提示词文件，支持动态加载。

| 文件 | 用途 |
|------|------|
| `main_prompt.txt` | 主提示词，定义Agent行为 |
| `rag_summarize.txt` | RAG总结提示词 |
| `report_prompt.txt` | 报告生成提示词 |
| `task_decompose.txt` | 任务拆解提示词 |
| `self_evaluation.txt` | 自我评估提示词 |
| `fact_check.txt` | 事实核查提示词 |

**提示词设计原则:**
- 清晰的角色定义
- 明确的任务指令
- 结构化的输出格式
- 容错能力

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Ollama（本地大模型）
- 依赖库见 `requirements.txt`

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd self_agent-dev1
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动 Ollama**
```bash
ollama run qwen3.5:9b
ollama run Mxbai-embed-large
```

### 运行方式

#### Streamlit 界面（推荐）
```bash
streamlit run app.py
```

#### FastAPI 后端服务
```bash
# 开发模式
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

#### Celery 任务队列（可选）
```bash
celery -A backend.tasks worker --loglevel=info
```

## ⚙️ 配置说明

### 主要配置文件

| 文件 | 说明 |
|------|------|
| `config/agent.yml` | Agent 配置（模型、高级能力开关、用户配置） |
| `config/chroma.yml` | Chroma 向量数据库配置 |
| `config/rag.yml` | RAG 检索配置（含混合检索权重） |
| `config/prompts.yml` | 提示词配置 |

### 混合检索权重配置

在 `config/rag.yml` 中设置：
```yaml
retrieval:
  bm25_weight: 0.5      # BM25检索权重
  vector_weight: 0.5    # 向量检索权重
  hybrid_weight: 0.5    # 混合权重
  rrf_k: 60             # RRF融合参数
```

### LangGraph 工作流配置

```yaml
langgraph:
  enabled: true          # 是否启用LangGraph工作流
  max_steps: 10         # 最大执行步骤（防止死循环）
  checkpoint: false      # 是否启用检查点（持久化状态）
  verbose: true         # 是否输出详细日志
```

## 📖 使用指南

### 知识库管理

1. 在左侧导航栏选择「知识库管理」
2. 创建新数据库或选择现有数据库
3. 上传文件（支持拖拽）
4. 查看文件列表、删除或重新解析文件

### 智能问答

1. 在「智能问答」页面输入问题
2. Agent 会自动分析问题类型：
   - 简单问题：直接检索知识库回答
   - 复杂问题：自动进行任务拆解和规划

### API 使用示例

**创建用户:**
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com"}'
```

**创建会话:**
```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-uuid", "name": "我的会话"}'
```

**聊天:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-uuid", "message": "你好", "mode": "react"}'
```

## 🛠️ 开发

### 添加新技能（Skills 框架）

1. 在 `agent/skills/` 目录下创建新技能文件夹
2. 创建 `SKILL.md` 技能说明文档
3. 创建 `reference/` 目录存放参考文档
4. 创建 `script/main.py` 实现技能逻辑
5. SkillLoader 会自动发现并加载新技能

### 添加传统工具

在 `agent/tools/` 目录下创建新的工具模块，然后在 `__init__.py` 中导出。

### 添加新提示词

在 `prompts/` 目录下创建新的提示词文件，并在 `config/prompts.yml` 中配置。

## 📝 提示词文件

| 文件 | 用途 |
|------|------|
| `main_prompt.txt` | 主提示词，定义 Agent 行为 |
| `rag_summarize.txt` | RAG 总结提示词 |
| `report_prompt.txt` | 报告生成提示词 |
| `task_decompose.txt` | 任务拆解提示词 |
| `self_evaluation.txt` | 自我评估提示词 |
| `fact_check.txt` | 事实核查提示词 |

## 📋 更新日志

### v1.5.0 (2026-04-20)

**新增功能:**
- ✅ FastAPI 后端接口
  - 用户管理 API（创建、查询、根据用户名获取）
  - 会话管理 API（创建、查询、删除）
  - 知识库 CRUD API（数据库列表、创建、删除、当前数据库信息）
  - 文件上传 API（同步/异步）
  - Agent 聊天 API
- ✅ 多用户会话隔离
  - 用户表和会话表（SQLite）
  - 会话数据隔离，不串流
  - 状态持久化支持（本地文件/数据库）
- ✅ 异步改造
  - 异步数据库操作
  - 异步 LLM 调用
  - Celery 任务队列（处理长任务）
  - 支持10人以内并发

**RAG模块优化:**
- ✅ 空Chunk过滤（长度<10的chunk被过滤）
- ✅ 大文件动态分块（根据文件大小自动调整chunk_size）
- ✅ Chroma缓存机制（提高检索性能）
- ✅ 混合检索权重可配置（BM25/向量/RRF权重）

**LangGraph工作流改进:**
- ✅ 最大步数限制（防止死循环）
- ✅ 异常捕获和失败重试机制
- ✅ 状态持久化（本地文件/数据库）
- ✅ 技能失败降级机制（FallbackSkill）

**修复问题:**
- ✅ 修复知识库参考文件路径显示临时文件路径的问题
- ✅ 修复 `OllamaEmbedding` 缺少 `embed_documents` 方法的问题
- ✅ 清理 git 追踪中的 `__pycache__` 和 `.vscode` 文件

### v1.4.1 (2026-04-17)

**修复问题:**
- ✅ 修复 `OllamaEmbedding` 缺少 `embed_query` 方法的问题
- ✅ 将默认 Embedding 模型从 `all-minilm` 改为 `Mxbai-embed-large`（支持1024维度）
- ✅ 修复数据库名称命名规范问题（确保3-512字符）
- ✅ 修复删除数据库时文件被占用问题（添加连接关闭和重试机制）
- ✅ 修复 `DuplicateWidgetID` 错误（添加数据库列表去重）
- ✅ 添加 `README.md` 到 `.gitignore`

**优化改进:**
- ✅ 添加 `_close_store()` 方法释放数据库连接
- ✅ 在 `delete_database()` 中添加连接关闭和重试逻辑
- ✅ 更新 `_sanitize_db_name()` 方法支持长度校验
- ✅ 更新 `_select_initial_database()` 过滤无效数据库名称

### v1.4.0 (2026-04-17)

**新增功能:**
- ✅ LangGraph 工作流引擎
  - 基于 LangGraph 重构 Agent 执行流程
  - 定义 5 个核心工作流节点：Think、Tool Call、Evaluate、Fact Check、Summarize
  - 实现状态管理（AgentState），支持消息历史、任务列表、工具结果等
  - 支持条件分支：根据评估结果决定继续执行或总结
  - 配置化工作流参数：最大步骤、检查点、详细日志

**工作流架构:**
```
用户输入 → Think → [工具调用?] → Tool Call → Evaluate → [继续?] → Summarize → 输出答案
                                              ↓
                                         Fact Check
```

**配置选项:**
```yaml
langgraph:
  enabled: true        # 是否启用LangGraph工作流
  max_steps: 10        # 最大执行步骤
  checkpoint: false    # 是否启用检查点（持久化状态）
  verbose: true        # 是否输出详细日志
```

### v1.3.0 (2026-04-16)

**新增功能:**
- ✅ 统一接口层（Unified API Abstraction）
  - 抽象基类：`BaseLLM`、`BaseEmbedding`、`BaseVisionLLM`
  - 工厂模式：`LLMFactory`、`EmbeddingFactory`、`VisionLLMFactory`
  - 支持多Provider切换：Azure OpenAI、OpenAI、DeepSeek、Ollama
  - 统一调用接口，上层代码无需关心底层实现

- ✅ 重排功能（Reranking）
  - 三种重排策略：Linear、Cross-Encoder、LLM
  - 自动降级机制：Cross-Encoder/LLM失败时自动降级到Linear
  - 集成到混合检索流程
  - 可配置重排方法和参数

- ✅ Ragas 量化评估功能
  - 检索效果评估：Hit Rate、MRR、Recall、Precision、Diversity
  - 生成效果评估：Faithfulness、Answer Relevance、Context Utilization、Answer Correctness、Answer Conciseness
  - Bad Case 分析：自动识别问题类型并提供优化建议
  - 支持分块策略、提示工程、召回逻辑的优化指导
  - 生成评估报告和优化建议

**检索流程优化:**
```
Query → Query Processing → Hybrid Search → RRF Fusion → Filtering → Reranking → Top-K Results
```

**评估流程:**
```
评估 → Bad Case 分析 → 优化建议 → 迭代改进
```

**Provider支持:**
| 提供者类型 | 典型场景 | 配置切换点 |
|------------|----------|------------|
| Azure OpenAI | 企业合规、私有云部署 | `provider: azure` |
| OpenAI 原生 | 通用开发、最新模型 | `provider: openai` |
| DeepSeek | 成本优化、特定语言优化 | `provider: deepseek` |
| Ollama/vLLM | 完全离线、隐私敏感 | `provider: ollama` |

### v1.2.0 (2026-04-16)

**新增功能:**
- ✅ 记忆管理系统（Memory Management）
  - 短期记忆：跟踪当前会话上下文，支持多轮对话
  - 长期记忆：通过 RAG 访问外部知识库，记住用户偏好和历史信息
- ✅ 意图识别能力（Intent Recognition）
  - 显式意图识别：规则匹配常见意图类型
  - 隐含意图分析：深度理解用户真实需求
  - 支持 6 种意图类型：知识查询、投诉分析、任务请求、数据检索、闲聊对话、偏好设置

**框架升级:**
- ✅ 引入 Anthropic Skills 标准框架
- ✅ 实现 SkillLoader 自动发现和加载机制
- ✅ 创建 6 个核心技能（rag_summarize、web_search、get_weather、task_decompose、fact_check、kb_management）
- ✅ 更新主提示词适配 Skills 框架和新功能

**代码清理:**
- ✅ 删除重复文件（agent_tools.py、core_tools.py）
- ✅ 移除重复工具函数（get_weather、rag_summarize、task_decompose、fact_check）
- ✅ 更新工具导出配置

**文档更新:**
- ✅ 更新 README 文档
- ✅ 为每个技能添加 SKILL.md 说明文档
- ✅ 添加参考文档目录

### v1.0.0 (2026-04-15)

**新增功能:**
- ✅ 知识库管理工具（统计、列表、创建、删除、切换数据库）
- ✅ 网络搜索功能（web_search、fetch_webpage）
- ✅ 高级 Agent 能力（任务拆解、路径规划、自我评估、事实核查、动态调整）
- ✅ 多文件格式支持（PDF、DOCX、TXT、MD、XLSX、CSV、HTML、图片）
- ✅ 多数据库管理（创建、选择、删除知识库）

**修复问题:**
- ✅ 修复 Chunk 为空的问题（分隔符配置错误）
- ✅ 修复上传完成后页面无限重新上传问题
- ✅ 修复删除文件时确认框弹回页面上方问题
- ✅ 修复 AttributeError: 'NoneType' object has no attribute 'get'
- ✅ 修复 agent 调用工具后无输出问题

**优化改进:**
- ✅ MD5文件移至数据库目录下
- ✅ 添加空Chunk过滤机制
- ✅ 统一提示词风格
- ✅ 添加测试文件整理
- ✅ 更新 .gitignore 配置

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**: 本项目为演示版本，生产环境使用请确保配置适当的安全措施和 API 密钥管理。
