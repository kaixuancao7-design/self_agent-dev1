# 开发与验证

建议在独立虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

`requirements.txt` 记录当前开发环境的直接依赖版本，并非完整锁文件；尚未验证全新环境安装。Firecrawl、Cross-Encoder 重排和 Unstructured 文档解析需要额外安装 `firecrawl-py`、`sentence-transformers`、`unstructured[md,xlsx]`。启用哪些能力就安装相应依赖。

运行前启动 Ollama，并准备 `config/agent.yml` 和 `config/rag.yml` 中指定的聊天及 Embedding 模型。

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
streamlit run app.py
```

API 模式会先创建用户再创建会话。知识库创建接口接收 JSON：`{"db_name": "my_kb"}`。异步上传还需要 Redis 与 Celery worker；worker 必须能访问 API 写入的临时文件路径，当前实现适合同机运行。

不依赖 Ollama、Redis 或真实知识库的回归测试：

```powershell
python -m pytest tests/test_core_regressions.py -q
```

测试使用真实 LangGraph 和 LangChain 工具对象，替换模型、数据库及外部服务。覆盖工作流执行、重试、消息持久化、API 请求格式、流式接口的可选用户检查与异步上传的知识库参数传递。

当前仍需处理：完整身份认证与授权、知识库按用户隔离、同会话并发控制、消息持久化，以及真实模型和任务队列的端到端验证。接口中的 `user_id` 是客户端传入值，不能视为可信身份。
