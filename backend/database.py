"""
数据库模型定义 - 使用 SQLite 和 SQLAlchemy

包含：
- User 表：用户信息
- Session 表：会话信息（与用户关联）
- Task 表：异步任务记录
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Boolean, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from uuid import uuid4

# 同步引擎（用于初始化）
sync_engine = create_engine("sqlite:///./backend/data/app.db", connect_args={"check_same_thread": False})

# 异步引擎
async_engine = create_async_engine("sqlite+aiosqlite:///./backend/data/app.db")

# 会话工厂
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# 基类
Base = declarative_base()


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255))
    password_hash = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关联会话
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class Session(Base):
    """会话表"""
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"))
    name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    message_count = Column(Integer, default=0)
    
    # 关联用户
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, name={self.name})>"


class Task(Base):
    """任务表（用于异步任务）"""
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"))
    session_id = Column(String(36), ForeignKey("sessions.id"))
    task_type = Column(String(50))  # file_upload, ragas_evaluation, document_parse
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)
    result = Column(Text)
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<Task(id={self.id}, type={self.task_type}, status={self.status})>"


# 初始化数据库
async def init_db():
    """异步初始化数据库"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# 依赖项：获取异步会话
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# 便捷函数
async def create_user(db: AsyncSession, username: str, email: str = None, password_hash: str = None) -> User:
    """创建用户"""
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        is_active=True
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    """根据用户名获取用户"""
    result = await db.execute(select(User).filter(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    """根据ID获取用户"""
    return await db.get(User, user_id)


async def create_session(db: AsyncSession, user_id: str, name: str = None) -> Session:
    """创建会话"""
    session = Session(
        user_id=user_id,
        name=name or f"会话 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        is_active=True,
        message_count=0
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    """获取会话"""
    return await db.get(Session, session_id)


async def get_user_sessions(db: AsyncSession, user_id: str) -> list[Session]:
    """获取用户的所有会话"""
    result = await db.execute(select(Session).filter(Session.user_id == user_id, Session.is_active == True))
    return result.scalars().all()


async def update_session_message_count(db: AsyncSession, session_id: str, increment: int = 1):
    """更新会话消息计数"""
    session = await db.get(Session, session_id)
    if session:
        session.message_count += increment
        await db.commit()


async def create_task(db: AsyncSession, user_id: str, task_type: str, session_id: str = None) -> Task:
    """创建任务"""
    task = Task(
        user_id=user_id,
        session_id=session_id,
        task_type=task_type,
        status="pending",
        progress=0
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task


async def update_task_status(db: AsyncSession, task_id: str, status: str, progress: int = None, result: str = None, error: str = None):
    """更新任务状态"""
    task = await db.get(Task, task_id)
    if task:
        task.status = status
        if progress is not None:
            task.progress = progress
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        await db.commit()


async def get_task(db: AsyncSession, task_id: str) -> Task | None:
    """获取任务"""
    return await db.get(Task, task_id)