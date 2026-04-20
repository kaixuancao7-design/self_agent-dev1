"""
FastAPI 后端服务 - 异步版本

提供以下 API：
- 用户管理（创建/登录/获取用户信息）
- 会话管理（创建/获取/删除会话，支持多用户隔离）
- 知识库操作（创建/删除/切换数据库，文件上传）
- Agent 调用（流式和非流式响应）
- 任务队列（文件处理、RAGAS评估等长任务）

技术特点：
- 异步数据库操作（SQLite + SQLAlchemy async）
- 用户与会话隔离
- 异步 LLM 调用（aiohttp）
- Celery 任务队列处理长任务
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Generator
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4
import json
from datetime import datetime

# 导入项目模块
from agent.react_agent import ReactAgent
from agent.langgraph_workflow import LangGraphAgent
from rag.vector_store import VectorStoreService
from utils.config_handler import agent_cfg, chroma_cfg

# 导入异步模块
from .database import (
    init_db, get_db,
    User, Session, Task,
    create_user, get_user_by_username, get_user_by_id,
    create_session, get_session, get_user_sessions,
    update_session_message_count, create_task, update_task_status, get_task
)
from .async_providers import AsyncOllamaProvider, AsyncRAGService, AsyncFileProcessor
from .tasks import process_large_file, evaluate_ragas, get_task_status

# 创建 FastAPI 应用
app = FastAPI(
    title="智能知识库助手 API",
    description="提供会话管理、知识库操作和 Agent 调用的 RESTful API（异步版本）",
    version="2.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态管理
vector_store = VectorStoreService()
async_ollama = AsyncOllamaProvider()

# Pydantic 模型定义
class UserCreate(BaseModel):
    """创建用户请求模型"""
    username: str
    email: Optional[str] = None
    password: Optional[str] = None

class UserResponse(BaseModel):
    """用户响应模型"""
    id: str
    username: str
    email: Optional[str]
    is_active: bool
    created_at: datetime

class SessionCreate(BaseModel):
    """创建会话请求模型"""
    user_id: str
    name: Optional[str] = None

class SessionResponse(BaseModel):
    """会话响应模型"""
    id: str
    user_id: str
    name: str
    message_count: int
    created_at: datetime

class ChatRequest(BaseModel):
    """聊天请求模型"""
    session_id: str
    message: str
    mode: Optional[str] = "react"  # react, langgraph, auto
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    """聊天响应模型（非流式）"""
    session_id: str
    response: str
    mode: str

class DatabaseResponse(BaseModel):
    """数据库操作响应模型"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    success: bool
    message: str
    md5: Optional[str] = None
    chunks: Optional[int] = None
    task_id: Optional[str] = None

class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    progress: int
    message: str
    data: Optional[Dict[str, Any]] = None

# 内存会话缓存（用于快速访问）
sessions_cache: Dict[str, Dict[str, Any]] = {}

def get_session_cache(session_id: str) -> Dict[str, Any]:
    """获取会话缓存，如果不存在则创建"""
    if session_id not in sessions_cache:
        sessions_cache[session_id] = {
            "session_id": session_id,
            "messages": [],
            "agent": None,
            "agent_mode": "react"
        }
    return sessions_cache[session_id]

# 事件处理
@app.on_event("startup")
async def startup_event():
    """启动时初始化数据库"""
    await init_db()

# 用户管理 API
@app.post("/api/users", response_model=UserResponse)
async def create_user_api(request: UserCreate, db=Depends(get_db)):
    """创建新用户"""
    # 检查用户名是否已存在
    existing_user = await get_user_by_username(db, request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 创建用户
    user = await create_user(
        db,
        username=request.username,
        email=request.email,
        password_hash=request.password  # 实际生产环境需要加密
    )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at
    )

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user_api(user_id: str, db=Depends(get_db)):
    """获取用户信息"""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at
    )

@app.get("/api/users/by-username/{username}", response_model=UserResponse)
async def get_user_by_username_api(username: str, db=Depends(get_db)):
    """根据用户名获取用户信息"""
    user = await get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at
    )

# 会话管理 API（支持多用户隔离）
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session_api(request: SessionCreate, db=Depends(get_db)):
    """创建新会话（与用户关联）"""
    # 验证用户存在
    user = await get_user_by_id(db, request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 创建会话
    session = await create_session(db, request.user_id, request.name)
    
    # 初始化会话缓存
    sessions_cache[session.id] = {
        "session_id": session.id,
        "user_id": request.user_id,
        "messages": [],
        "agent": None,
        "agent_mode": "react"
    }
    
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        name=session.name,
        message_count=session.message_count,
        created_at=session.created_at
    )

@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session_api(session_id: str, db=Depends(get_db)):
    """获取会话信息"""
    session = await get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return SessionResponse(
        id=session.id,
        user_id=session.user_id,
        name=session.name,
        message_count=session.message_count,
        created_at=session.created_at
    )

@app.delete("/api/sessions/{session_id}", response_model=DatabaseResponse)
async def delete_session_api(session_id: str, db=Depends(get_db)):
    """删除会话"""
    session = await get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 删除数据库中的会话
    await db.delete(session)
    await db.commit()
    
    # 删除缓存
    if session_id in sessions_cache:
        del sessions_cache[session_id]
    
    return DatabaseResponse(
        success=True,
        message=f"会话 {session_id} 删除成功"
    )

@app.get("/api/users/{user_id}/sessions", response_model=List[SessionResponse])
async def get_user_sessions_api(user_id: str, db=Depends(get_db)):
    """获取用户的所有会话"""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    sessions = await get_user_sessions(db, user_id)
    
    return [
        SessionResponse(
            id=s.id,
            user_id=s.user_id,
            name=s.name,
            message_count=s.message_count,
            created_at=s.created_at
        )
        for s in sessions
    ]

# Agent 调用 API（异步版本）
@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest, db=Depends(get_db)):
    """异步非流式聊天接口"""
    # 验证会话存在
    session = await get_session(db, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 验证用户权限（可选）
    if request.user_id and session.user_id != request.user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    
    # 获取会话缓存
    session_cache = get_session_cache(request.session_id)
    
    # 确定模式
    mode = request.mode.lower()
    if mode == "auto":
        if len(request.message) > 50 or any(k in request.message for k in ["帮我", "请", "制定", "规划", "创建"]):
            mode = "langgraph"
        else:
            mode = "react"
    
    # 创建或切换 Agent
    if session_cache["agent_mode"] != mode or session_cache["agent"] is None:
        if mode == "react":
            session_cache["agent"] = ReactAgent()
        else:
            session_cache["agent"] = LangGraphAgent()
        session_cache["agent_mode"] = mode
    
    # 执行 Agent（同步调用包装为异步）
    import asyncio
    loop = asyncio.get_event_loop()
    
    if mode == "react":
        response = await loop.run_in_executor(None, session_cache["agent"].stream_to_text, request.message)
    else:
        result = await loop.run_in_executor(None, session_cache["agent"].run, request.message)
        response = result.get("final_answer", "处理完成")
    
    # 保存消息
    session_cache["messages"].append({"role": "user", "content": request.message})
    session_cache["messages"].append({"role": "assistant", "content": response})
    
    # 更新数据库中的消息计数
    await update_session_message_count(db, request.session_id)
    
    return ChatResponse(
        session_id=request.session_id,
        response=response,
        mode=mode
    )

@app.post("/api/chat/stream")
async def chat_stream_api(request: ChatRequest, db=Depends(get_db)):
    """异步流式聊天接口"""
    # 验证会话存在
    session = await get_session(db, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 获取会话缓存
    session_cache = get_session_cache(request.session_id)
    
    # 确定模式
    mode = request.mode.lower()
    if mode == "auto":
        if len(request.message) > 50 or any(k in request.message for k in ["帮我", "请", "制定", "规划", "创建"]):
            mode = "langgraph"
        else:
            mode = "react"
    
    # 创建或切换 Agent
    if session_cache["agent_mode"] != mode or session_cache["agent"] is None:
        if mode == "react":
            session_cache["agent"] = ReactAgent()
        else:
            session_cache["agent"] = LangGraphAgent()
        session_cache["agent_mode"] = mode
    
    # 保存用户消息
    session_cache["messages"].append({"role": "user", "content": request.message})
    
    # 定义生成器（使用 async generator）
    async def generate_response():
        full_response = ""
        
        if mode == "react":
            for chunk in session_cache["agent"].execute_stream(request.message):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk, 'mode': mode})}\n\n"
        else:
            for chunk in session_cache["agent"].run_stream(request.message):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk, 'mode': mode})}\n\n"
        
        # 保存完整响应
        session_cache["messages"].append({"role": "assistant", "content": full_response})
        
        # 更新数据库中的消息计数
        await update_session_message_count(db, request.session_id)
        
        yield f"data: {json.dumps({'chunk': '', 'mode': mode, 'finished': True})}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/event-stream")

# 知识库操作 API
@app.get("/api/databases", response_model=List[str])
async def list_databases_api():
    """获取数据库列表"""
    return VectorStoreService.list_databases()

@app.post("/api/databases", response_model=DatabaseResponse)
async def create_database_api(db_name: str):
    """创建新数据库"""
    result = vector_store.create_database(db_name)
    return DatabaseResponse(
        success=result["success"],
        message=result["message"],
        data={"db_name": db_name}
    )

@app.delete("/api/databases/{db_name}", response_model=DatabaseResponse)
async def delete_database_api(db_name: str):
    """删除数据库"""
    current_db = vector_store.current_db
    if current_db == db_name:
        db_list = VectorStoreService.list_databases()
        if len(db_list) <= 1:
            return DatabaseResponse(
                success=False,
                message="至少需要保留一个数据库"
            )
        new_db = [db for db in db_list if db != db_name][0]
        vector_store.switch_database(new_db)
    
    result = vector_store.delete_database(db_name)
    return DatabaseResponse(
        success=result["success"],
        message=result["message"]
    )

@app.post("/api/databases/{db_name}/switch", response_model=DatabaseResponse)
async def switch_database_api(db_name: str):
    """切换数据库"""
    result = vector_store.switch_database(db_name)
    return DatabaseResponse(
        success=result["success"],
        message=result["message"],
        data={"current_db": vector_store.current_db}
    )

@app.get("/api/databases/current", response_model=Dict[str, Any])
async def get_current_database_api():
    """获取当前数据库信息"""
    stats = vector_store.get_collection_stats()
    return {
        "current_db": vector_store.current_db,
        **stats
    }

@app.get("/api/databases/stats", response_model=Dict[str, Any])
async def get_database_stats_api():
    """获取数据库统计信息"""
    return vector_store.get_collection_stats()

# 文件上传 API（支持异步任务队列）
@app.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file_api(file: UploadFile = File(...), user_id: Optional[str] = None):
    """上传文件到知识库（同步方式）"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        result = vector_store.upload_file(temp_file_path, file.filename)
        return FileUploadResponse(
            success=result["success"],
            message=result["message"],
            md5=result.get("md5"),
            chunks=result.get("chunks")
        )
    finally:
        os.unlink(temp_file_path)

@app.post("/api/files/upload/async", response_model=FileUploadResponse)
async def upload_file_async_api(file: UploadFile = File(...), user_id: Optional[str] = None, db=Depends(get_db)):
    """上传文件到知识库（异步方式，使用 Celery 任务队列）"""
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    # 创建任务记录
    task = await create_task(db, user_id or "anonymous", "file_upload")
    
    # 提交 Celery 任务
    celery_task = process_large_file.apply_async(
        args=(temp_file_path, file.filename),
        task_id=task.id
    )
    
    return FileUploadResponse(
        success=True,
        message="文件上传任务已提交，正在处理中",
        task_id=task.id
    )

@app.get("/api/files", response_model=List[Dict[str, Any]])
async def list_files_api():
    """获取已上传文件列表"""
    return vector_store.get_uploaded_files()

@app.delete("/api/files/{md5}", response_model=DatabaseResponse)
async def delete_file_api(md5: str):
    """删除文件"""
    result = vector_store.remove_file(md5)
    return DatabaseResponse(
        success=result["success"],
        message=result["message"]
    )

@app.post("/api/files/{md5}/reparse", response_model=FileUploadResponse)
async def reparse_file_api(md5: str, file: UploadFile = File(...)):
    """重新解析文件"""
    from utils.file_handler import get_file_md5_hex
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        file_md5 = get_file_md5_hex(temp_file_path)
        if file_md5 != md5:
            return FileUploadResponse(
                success=False,
                message="上传文件的 MD5 与目标文件不匹配"
            )
        
        result = vector_store.reparse_file(temp_file_path, file.filename)
        return FileUploadResponse(
            success=result["success"],
            message=result["message"],
            md5=result.get("md5"),
            chunks=result.get("chunks")
        )
    finally:
        os.unlink(temp_file_path)

# 任务队列 API
@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status_api(task_id: str):
    """获取任务状态"""
    status = get_task_status(task_id)
    return TaskResponse(
        task_id=status["task_id"],
        status=status["status"],
        progress=status["progress"],
        message=status["message"],
        data=status.get("data")
    )

@app.post("/api/tasks/evaluate", response_model=TaskResponse)
async def create_evaluate_task_api(query: str, answer: str, contexts: List[str], user_id: Optional[str] = None, db=Depends(get_db)):
    """创建 RAGAS 评估任务"""
    # 创建任务记录
    task = await create_task(db, user_id or "anonymous", "ragas_evaluation")
    
    # 提交 Celery 任务
    celery_task = evaluate_ragas.apply_async(
        args=(query, answer, contexts),
        task_id=task.id
    )
    
    return TaskResponse(
        task_id=task.id,
        status="pending",
        progress=0,
        message="评估任务已提交"
    )

# 健康检查
@app.get("/api/health")
async def health_check_api():
    """健康检查接口"""
    return {"status": "healthy", "service": "smart-knowledge-assistant", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)