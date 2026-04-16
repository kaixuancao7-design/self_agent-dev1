"""
记忆管理模块

实现 Agent 的记忆功能：
- 短期记忆：当前会话上下文管理
- 长期记忆：通过 RAG 访问外部知识库，记住用户偏好和历史信息
"""
from typing import List, Dict, Any
from datetime import datetime
from utils.logger_handler import logger


class ShortTermMemory:
    """短期记忆 - 跟踪当前会话的上下文"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_length = 20  # 最大历史记录数
    
    def add_message(self, role: str, content: str, timestamp: datetime = None):
        """
        添加消息到短期记忆
        
        :param role: 角色（user/assistant/system）
        :param content: 消息内容
        :param timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat()
        }
        
        self.conversation_history.append(message)
        
        # 保持历史记录在最大长度内
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        logger.info(f"[ShortTermMemory] 添加消息: {role} - {content[:50]}...")
    
    def get_history(self, recent_n: int = None) -> List[Dict[str, Any]]:
        """
        获取历史记录
        
        :param recent_n: 获取最近的n条记录，默认全部
        :return: 历史记录列表
        """
        if recent_n is None:
            return self.conversation_history
        return self.conversation_history[-recent_n:]
    
    def get_context_summary(self) -> str:
        """
        获取当前会话上下文摘要
        
        :return: 上下文摘要字符串
        """
        if not self.conversation_history:
            return "无历史对话记录"
        
        summary = "当前会话摘要：\n"
        for msg in self.conversation_history[-5:]:  # 最近5条
            summary += f"- [{msg['role']}] {msg['content'][:80]}...\n"
        
        return summary
    
    def clear(self):
        """清空短期记忆"""
        self.conversation_history = []
        logger.info("[ShortTermMemory] 已清空短期记忆")
    
    def get_length(self) -> int:
        """获取历史记录长度"""
        return len(self.conversation_history)


class LongTermMemory:
    """长期记忆 - 通过 RAG 访问外部知识库"""
    
    def __init__(self):
        self.user_preferences: Dict[str, Any] = {}
        self.task_progress: Dict[str, Any] = {}
        self.business_knowledge: List[Dict[str, Any]] = []
    
    def load_user_preferences(self, user_id: str):
        """
        加载用户偏好
        
        :param user_id: 用户ID
        """
        # 从知识库检索用户偏好
        from rag.rag_service import RagSummerizeService
        
        try:
            rag_service = RagSummerizeService()
            query = f"用户 {user_id} 的偏好设置"
            result = rag_service.rag_summarize(query)
            
            # 解析偏好信息
            if "偏好" in result or "设置" in result:
                self.user_preferences[user_id] = {
                    "raw_data": result,
                    "last_updated": datetime.now().isoformat()
                }
                logger.info(f"[LongTermMemory] 已加载用户 {user_id} 的偏好")
        except Exception as e:
            logger.error(f"[LongTermMemory] 加载用户偏好失败: {e}")
    
    def save_user_preference(self, user_id: str, key: str, value: Any):
        """
        保存用户偏好
        
        :param user_id: 用户ID
        :param key: 偏好键
        :param value: 偏好值
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {"preferences": {}}
        
        self.user_preferences[user_id]["preferences"][key] = value
        self.user_preferences[user_id]["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"[LongTermMemory] 保存用户 {user_id} 的偏好: {key} = {value}")
    
    def get_user_preference(self, user_id: str, key: str = None) -> Any:
        """
        获取用户偏好
        
        :param user_id: 用户ID
        :param key: 偏好键（可选，不传则返回所有偏好）
        :return: 偏好值
        """
        if user_id not in self.user_preferences:
            self.load_user_preferences(user_id)
        
        if user_id in self.user_preferences:
            if key:
                return self.user_preferences[user_id].get("preferences", {}).get(key)
            return self.user_preferences[user_id].get("preferences", {})
        
        return None
    
    def update_task_progress(self, task_id: str, status: str, progress: float = 0.0, details: str = ""):
        """
        更新任务进度
        
        :param task_id: 任务ID
        :param status: 任务状态（pending/in_progress/completed/failed）
        :param progress: 进度百分比（0-1）
        :param details: 详细信息
        """
        self.task_progress[task_id] = {
            "status": status,
            "progress": progress,
            "details": details,
            "updated_at": datetime.now().isoformat()
        }
        logger.info(f"[LongTermMemory] 更新任务进度: {task_id} - {status} ({progress*100}%)")
    
    def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务进度
        
        :param task_id: 任务ID
        :return: 任务进度信息
        """
        return self.task_progress.get(task_id, {})
    
    def retrieve_business_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        检索业务知识
        
        :param query: 检索词
        :return: 相关知识列表
        """
        from rag.rag_service import RagSummerizeService
        
        try:
            rag_service = RagSummerizeService()
            result = rag_service.rag_summarize(query)
            
            # 解析结果
            knowledge_items = []
            if result:
                knowledge_items.append({
                    "content": result,
                    "source": "knowledge_base",
                    "retrieved_at": datetime.now().isoformat()
                })
            
            logger.info(f"[LongTermMemory] 检索到 {len(knowledge_items)} 条业务知识")
            return knowledge_items
        except Exception as e:
            logger.error(f"[LongTermMemory] 检索业务知识失败: {e}")
            return []


class MemoryManager:
    """记忆管理器 - 整合短期和长期记忆"""
    
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.current_user_id = None
    
    def set_current_user(self, user_id: str):
        """
        设置当前用户
        
        :param user_id: 用户ID
        """
        self.current_user_id = user_id
        # 预加载用户偏好
        self.long_term.load_user_preferences(user_id)
        logger.info(f"[MemoryManager] 当前用户已设置: {user_id}")
    
    def add_conversation(self, role: str, content: str):
        """
        添加对话到短期记忆
        
        :param role: 角色
        :param content: 内容
        """
        self.short_term.add_message(role, content)
    
    def get_session_context(self) -> str:
        """
        获取当前会话上下文（包含短期和长期记忆）
        
        :return: 上下文字符串
        """
        context = ""
        
        # 添加短期记忆
        short_term_context = self.short_term.get_context_summary()
        if short_term_context:
            context += f"{short_term_context}\n"
        
        # 添加长期记忆（用户偏好）
        if self.current_user_id:
            preferences = self.long_term.get_user_preference(self.current_user_id)
            if preferences:
                context += f"\n用户偏好：\n{preferences}\n"
        
        return context.strip()
    
    def retrieve_relevant_memory(self, query: str) -> str:
        """
        检索与当前查询相关的记忆
        
        :param query: 用户查询
        :return: 相关记忆内容
        """
        # 1. 检查短期记忆中是否有相关内容
        relevant_history = []
        for msg in self.short_term.get_history():
            if any(keyword in msg["content"] for keyword in query.split()[:5]):
                relevant_history.append(msg)
        
        # 2. 从长期记忆检索业务知识
        business_knowledge = self.long_term.retrieve_business_knowledge(query)
        
        # 3. 构建结果
        result = ""
        
        if relevant_history:
            result += "【上下文参考】\n"
            for msg in relevant_history[-3:]:
                result += f"- [{msg['role']}] {msg['content']}\n"
        
        if business_knowledge:
            result += "\n【知识库参考】\n"
            for item in business_knowledge:
                result += f"{item['content'][:500]}...\n"
        
        return result
    
    def clear_session(self):
        """清除当前会话记忆"""
        self.short_term.clear()
        logger.info("[MemoryManager] 会话记忆已清除")


# 全局记忆管理器实例
memory_manager = MemoryManager()
