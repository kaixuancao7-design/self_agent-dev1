"""Core workflow regressions, using real LangGraph and tools without external services."""
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


ROOT = Path(__file__).resolve().parents[1]


def stub(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def workflow(tmp_path):
    model = SimpleNamespace(chat=Mock(return_value="final answer"))
    loader = SimpleNamespace(get_tool_list=Mock(return_value=[]))
    modules = {
        "model.factory": stub("model.factory", chat_model=model),
        "utils.logger_handler": stub("utils.logger_handler", logger=logging.getLogger("test")),
        "rag.ragas_evaluator": stub("rag.ragas_evaluator", evaluate_rag_pipeline=Mock()),
        "backend.database": stub("backend.database", AsyncSessionLocal=Mock(), WorkflowState=Mock()),
        "agent.skills": stub("agent.skills", skill_loader=loader),
    }
    spec = importlib.util.spec_from_file_location("workflow_under_test", ROOT / "agent/langgraph_workflow.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        yield module, loader


def test_default_tools_and_explicit_empty_list(workflow, tmp_path):
    module, loader = workflow
    loader.get_tool_list.return_value = ["loaded tool"]
    assert module.LangGraphAgent(persist_dir=str(tmp_path)).tools == ["loaded tool"]
    assert module.LangGraphAgent(tools=[], persist_dir=str(tmp_path)).tools == []


def test_knowledge_query_runs_tool_and_summarizes(workflow, tmp_path):
    module, _ = workflow
    calls = []

    @tool
    def rag_summarize(query: str) -> str:
        """Retrieve reference material."""
        calls.append(query)
        return "success: reference text"

    agent = module.LangGraphAgent(tools=[rag_summarize], persist_dir=str(tmp_path))
    result = agent.run("什么是知识库")
    assert calls == ["什么是知识库"]
    assert result["final_answer"] == "final answer"
    assert result["is_finished"]
    assert any(isinstance(message, ToolMessage) for message in result["messages"])


def test_retry_preserves_tool_arguments(workflow, tmp_path):
    module, _ = workflow
    calls = []

    @tool
    def task_decompose(goal: str) -> str:
        """Decompose a goal."""
        calls.append(goal)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return "success: planned"

    agent = module.LangGraphAgent(tools=[task_decompose], retry_delay=0, persist_dir=str(tmp_path))
    result = agent.run("帮我制定计划")
    assert calls == ["帮我制定计划"] * 2
    assert result["last_error"] is None
    assert result["final_answer"] == "final answer"


def test_exhausted_retries_produce_failure_answer(workflow, tmp_path):
    module, _ = workflow
    calls = []

    @tool
    def rag_summarize(query: str) -> str:
        """Retrieve reference material."""
        calls.append(query)
        raise RuntimeError("unavailable")

    agent = module.LangGraphAgent(tools=[rag_summarize], max_retries=2, retry_delay=0, persist_dir=str(tmp_path))
    result = agent.run("什么是知识库")
    assert len(calls) == 2
    assert "unavailable" in result["final_answer"]
    assert result["is_finished"]


def test_state_file_round_trip_preserves_messages(workflow, tmp_path):
    module, _ = workflow
    agent = module.LangGraphAgent(tools=[], persist_dir=str(tmp_path))
    state = agent.run("你好")
    agent._save_state_to_file("session", state)
    loaded = agent._load_state_from_file("session")
    assert isinstance(loaded["messages"][0], HumanMessage)
    assert loaded == state


def test_step_limit_still_returns_answer(workflow, tmp_path):
    module, _ = workflow
    agent = module.LangGraphAgent(tools=[], max_steps=1, persist_dir=str(tmp_path))
    assert agent.run("你好")["final_answer"] == "final answer"


def test_low_quality_results_reach_step_limit_without_recursion_error(workflow, tmp_path):
    module, _ = workflow

    @tool
    def rag_summarize(query: str) -> str:
        """Retrieve reference material."""
        return "short"

    agent = module.LangGraphAgent(tools=[rag_summarize], max_steps=9, persist_dir=str(tmp_path))
    result = agent.run("什么是知识库")
    assert result["step_count"] == 9
    assert result["final_answer"] == "final answer"


@pytest.fixture
def api():
    async def get_db():
        yield Mock()

    database = stub("backend.database", get_db=get_db)
    for name in ("init_db", "create_user", "get_user_by_username", "get_user_by_id",
                 "create_session", "get_session", "get_user_sessions", "update_session_message_count",
                 "create_task", "update_task_status", "get_task"):
        setattr(database, name, AsyncMock())
    for name in ("User", "Session", "Task"):
        setattr(database, name, Mock())
    store = Mock(current_db="selected_kb")
    store.create_database.return_value = {"success": True, "message": "created"}
    vector_factory = Mock(return_value=store)
    task = Mock()
    modules = {
        "agent.react_agent": stub("agent.react_agent", ReactAgent=Mock()),
        "agent.langgraph_workflow": stub("agent.langgraph_workflow", LangGraphAgent=Mock()),
        "rag.vector_store": stub("rag.vector_store", VectorStoreService=vector_factory),
        "utils.config_handler": stub("utils.config_handler", agent_cfg={}, chroma_cfg={}),
        "backend.database": database,
        "backend.async_providers": stub("backend.async_providers", AsyncOllamaProvider=Mock(), AsyncRAGService=Mock(), AsyncFileProcessor=Mock()),
        "backend.tasks": stub("backend.tasks", process_large_file=task, evaluate_ragas=Mock(), get_task_status=Mock()),
    }
    spec = importlib.util.spec_from_file_location("backend.main_under_test", ROOT / "backend/main.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
        yield module, database, store, task


def complete_immediately(coroutine):
    """Run handlers whose mocked async dependencies complete without I/O."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    finally:
        coroutine.close()
    raise AssertionError("Unexpected external async I/O")


def test_create_database_accepts_frontend_json(api):
    module, _, store, _ = api
    request_body = module.app.openapi()["paths"]["/api/databases"]["post"]["requestBody"]
    assert "application/json" in request_body["content"]
    response = complete_immediately(module.create_database_api(db_name="new_kb"))
    assert response.success
    store.create_database.assert_called_once_with("new_kb")


def test_stream_checks_supplied_owner(api):
    from fastapi import HTTPException
    module, database, _, _ = api
    database.get_session.return_value = SimpleNamespace(user_id="owner")
    request = module.ChatRequest(session_id="session", message="hello", user_id="other")
    with pytest.raises(HTTPException) as error:
        complete_immediately(module.chat_stream_api(request, db=Mock()))
    assert error.value.status_code == 403


def test_async_upload_keeps_selected_database(api):
    module, database, _, task = api
    database.create_task.return_value = SimpleNamespace(id="task-id")
    upload = SimpleNamespace(filename="note.txt", read=AsyncMock(return_value=b"document"))
    response = complete_immediately(module.upload_file_async_api(file=upload, db=Mock()))
    assert response.success
    file_path, file_name, config = task.apply_async.call_args.kwargs["args"]
    try:
        assert file_name == "note.txt"
        assert config == {"db_name": "selected_kb"}
        assert Path(file_path).read_bytes() == b"document"
    finally:
        Path(file_path).unlink()


def test_celery_upload_uses_existing_vector_store(tmp_path):
    store_factory = Mock()
    store_factory.return_value.upload_file.return_value = {"success": True}
    spec = importlib.util.spec_from_file_location("tasks_under_test", ROOT / "backend/tasks.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "rag.ragas_evaluator": stub("rag.ragas_evaluator", evaluate_rag_pipeline=Mock()),
        "rag.vector_store": stub("rag.vector_store", VectorStoreService=store_factory),
    }):
        spec.loader.exec_module(module)
        task_function = module.process_large_file.run.__func__
        result = task_function(Mock(), "file.txt", "file.txt", {"db_name": "selected_kb"})
    store_factory.assert_called_once_with(db_name="selected_kb")
    assert result["status"] == "success"
