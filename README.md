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
- 图片内容索引与检索

### 🤖 高级 Agent 能力

| 能力 | 描述 |
|------|------|
| **任务拆解** | 将复杂目标自动拆解为可执行的子任务序列 |
| **路径规划** | 为子任务排序，确定执行逻辑和优先级 |
| **自我评估** | 评估每步结果质量，自动调整策略 |
| **事实核查** | 校验答案与参考资料一致性，避免幻觉 |
| **动态调整** | 处理失败后的备选方案 |

### 🔧 工具系统
- **核心工具**: RAG 总结服务
- **网络搜索**: 实时资讯获取、网页内容抓取
- **通用工具**: 天气查询、用户定位、时间获取、知识库管理
- **报告工具**: 用户历史记录查询、报告生成
- **高级工具**: 任务拆解、路径规划、自我评估、事实核查、动态调整

## 🏗️ 项目结构

```
├── agent/                    # Agent 核心模块
│   ├── modules/             # 高级能力模块
│   │   ├── task_decomposer.py      # 任务拆解
│   │   ├── path_planner.py        # 路径规划
│   │   ├── self_evaluator.py      # 自我评估
│   │   ├── fact_checker.py        # 事实核查
│   │   └── dynamic_adjuster.py    # 动态调整
│   ├── tools/               # 工具模块
│   │   ├── core_tools.py          # 核心业务工具
│   │   ├── utility_tools.py       # 通用工具
│   │   ├── report_tools.py        # 报告工具
│   │   ├── advanced_tools.py      # 高级能力工具
│   │   └── search_tools.py        # 网络搜索工具
│   ├── react_agent.py       # 主 Agent 实现
│   └── session.py           # 会话管理
├── rag/                     # RAG 检索服务
│   ├── vector_store.py      # 向量存储
│   ├── bm25_index.py        # BM25 索引
│   ├── hybrid_retriever.py  # 混合检索
│   ├── image_index.py       # 图片索引
│   └── rag_service.py       # RAG 服务
├── config/                  # 配置文件
├── prompts/                 # 提示词文件
├── chromadb/                # Chroma 向量数据库
├── data/                    # 数据文件
├── utils/                   # 工具函数
├── model/                   # 模型工厂
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

## 🔧 工具列表

| 工具名称 | 功能 | 参数 |
|---------|------|------|
| `rag_summarize` | 知识库检索 | query |
| `web_search` | 网络搜索 | query, max_results |
| `fetch_webpage` | 获取网页内容 | url |
| `get_weather` | 获取天气 | city |
| `get_user_location` | 获取用户位置 | 无 |
| `get_user_id` | 获取用户ID | 无 |
| `get_current_month` | 获取当前月份 | 无 |
| `task_decompose` | 任务拆解 | goal |
| `evaluate_result` | 结果评估 | task_description, result |
| `fact_check` | 事实核查 | answer, sources |

## 🛠️ 开发

### 添加新工具

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
