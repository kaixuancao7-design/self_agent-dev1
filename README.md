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

## 🏗️ 项目结构

```
├── agent/                    # Agent 核心模块
│   ├── modules/             # 高级能力模块
│   │   ├── task_decomposer.py      # 任务拆解
│   │   ├── path_planner.py        # 路径规划
│   │   ├── self_evaluator.py      # 自我评估
│   │   ├── fact_checker.py        # 事实核查
│   │   ├── dynamic_adjuster.py    # 动态调整
│   │   ├── memory_manager.py      # 记忆管理（短期+长期记忆）
│   │   └── intent_recognizer.py   # 意图识别
│   ├── skills/              # Skills 框架（Anthropic 标准）
│   │   ├── __init__.py            # 技能导出
│   │   ├── skill_loader.py        # 技能加载器（自动发现/加载）
│   │   ├── rag_summarize/         # RAG检索技能
│   │   │   ├── SKILL.md           # 技能说明文档
│   │   │   ├── reference/         # 参考文档
│   │   │   └── script/main.py     # 可执行脚本
│   │   ├── web_search/            # 网络搜索技能
│   │   ├── get_weather/           # 天气查询技能
│   │   ├── task_decompose/        # 任务拆解技能
│   │   ├── fact_check/            # 事实核查技能
│   │   ├── kb_management/         # 知识库管理技能
│   │   ├── content_generator/     # 智能内容生成技能
│   │   ├── document_processor/    # 文档处理技能
│   │   └── knowledge_governance/  # 知识治理技能
│   ├── tools/               # 传统工具模块（细粒度工具）
│   │   ├── __init__.py            # 工具导出
│   │   ├── utility_tools.py       # 通用工具+知识库管理
│   │   ├── report_tools.py        # 报告工具
│   │   ├── advanced_tools.py      # 高级能力工具（评估）
│   │   └── search_tools.py        # 网络搜索工具
│   ├── react_agent.py       # 主 Agent 实现（集成Skills框架）
│   └── session.py           # 会话管理
├── model/                   # 统一接口层（Unified API Abstraction）
│   ├── __init__.py          # 模块导出
│   ├── base.py              # 抽象基类（BaseLLM、BaseEmbedding、BaseVisionLLM）
│   ├── factory.py           # 工厂类（LLMFactory、EmbeddingFactory、VisionLLMFactory）
│   └── providers/           # Provider实现
│       ├── __init__.py              # Provider导出
│       ├── azure_provider.py        # Azure OpenAI实现
│       ├── openai_provider.py       # OpenAI原生实现
│       ├── deepseek_provider.py     # DeepSeek实现
│       └── ollama_provider.py       # Ollama本地实现
├── rag/                     # RAG 检索服务
│   ├── vector_store.py      # 向量存储（含空Chunk过滤）
│   ├── bm25_index.py        # BM25 索引
│   ├── hybrid_retriever.py  # 混合检索（Query Processing、RRF Fusion）
│   ├── reranker.py          # 重排器（Linear/Cross-Encoder/LLM）
│   ├── ragas_evaluator.py   # Ragas 量化评估器
│   ├── image_index.py       # 图片索引
│   └── rag_service.py       # RAG 服务
├── scripts/                 # 运行脚本
│   └── run_ragas_evaluation.py  # Ragas 评估脚本
├── config/                  # 配置文件
│   ├── agent.yml            # Agent 配置
│   ├── chroma.yml           # Chroma 配置
│   ├── rag.yml              # RAG 配置
│   └── prompts.yml          # 提示词配置
├── prompts/                 # 提示词文件
│   ├── main_prompt.txt      # 主提示词
│   ├── rag_summarize.txt    # RAG总结提示词
│   ├── report_prompt.txt    # 报告生成提示词
│   ├── task_decompose.txt   # 任务拆解提示词
│   ├── self_evaluation.txt  # 自我评估提示词
│   └── fact_check.txt       # 事实核查提示词
├── chromadb/                # Chroma 向量数据库（按数据库名组织）
├── data/                    # 数据文件（按数据库名组织）
├── utils/                   # 工具函数
├── model/                   # 模型工厂
├── test/                    # 测试文件
│   └── test_skills_framework.py  # Skills框架测试
└── app.py                   # Streamlit 应用入口
```

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
```

4. **运行应用**
```bash
streamlit run app.py
```

## ⚙️ 配置说明

### 主要配置文件

| 文件 | 说明 |
|------|------|
| `config/agent.yml` | Agent 配置（模型、高级能力开关、用户配置） |
| `config/chroma.yml` | Chroma 向量数据库配置 |
| `config/rag.yml` | RAG 检索配置 |
| `config/prompts.yml` | 提示词配置 |

### 高级能力开关

在 `config/agent.yml` 中设置：
```yaml
enable_advanced_features: true  # 启用高级能力
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

### 报告生成

输入类似「生成我的使用报告」的指令，系统会自动生成个性化报告。

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
