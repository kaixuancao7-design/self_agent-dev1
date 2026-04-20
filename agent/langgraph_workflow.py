"""
LangGraph 工作流架构

基于 LangGraph 重构的 Agent 工作流，包含：
- 思考节点：分析用户意图，决定下一步行动
- 工具调用节点：执行工具操作（带重试机制）
- 总结节点：生成最终回复
- 评估节点：评估执行结果质量
- 事实核查节点：校验答案与参考资料一致性

工作流设计遵循 ReAct 模式：思考 → 行动 → 观察 → 总结

新增特性：
- 最大步数限制：防止死循环
- 异常捕获：增强系统稳定性
- 失败重试：基础版重试机制
- 状态持久化：支持本地文件和数据库存储
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from utils.logger_handler import logger
from model.factory import chat_model
from rag.ragas_evaluator import evaluate_rag_pipeline

# 尝试导入数据库相关模块
try:
    from database.models import WorkflowState, SessionLocal
    from sqlalchemy import func
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("[LangGraphAgent] 数据库模块不可用，状态持久化将使用本地文件")


class AgentState(TypedDict):
    """
    Agent 工作流状态定义
    
    Attributes:
        messages: 消息历史（包含用户输入、工具调用、工具结果等）
        current_task: 当前任务描述
        task_list: 待执行任务列表（用于多步任务）
        tool_results: 工具执行结果
        is_finished: 是否完成
        final_answer: 最终答案
        evaluation_score: 评估分数
        step_count: 当前执行步数
        session_id: 会话ID（用于状态持久化）
        retry_count: 当前重试次数
        last_error: 最后一次错误信息
    """
    messages: List[Dict[str, Any]]
    current_task: Optional[str] = None
    task_list: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    is_finished: bool = False
    final_answer: Optional[str] = None
    evaluation_score: Optional[float] = None
    step_count: int = 0
    session_id: Optional[str] = None
    retry_count: int = 0
    last_error: Optional[str] = None


class LangGraphAgent:
    """基于 LangGraph 的 Agent 工作流，支持最大步数限制、异常捕获和状态持久化"""
    
    def __init__(self, tools: List[BaseTool] = None, max_steps: int = 10, 
                 max_retries: int = 3, retry_delay: float = 1.0,
                 persist_dir: str = "workflow_states"):
        """
        初始化 LangGraph Agent
        
        :param tools: 可用工具列表
        :param max_steps: 最大步数限制（防止死循环）
        :param max_retries: 最大重试次数
        :param retry_delay: 重试延迟（秒）
        :param persist_dir: 状态持久化目录（本地文件模式）
        """
        self.tools = tools or []
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.persist_dir = persist_dir
        
        # 创建持久化目录
        os.makedirs(persist_dir, exist_ok=True)
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("think", self._think_node)
        graph.add_node("tool_call", self._tool_call_node)
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("fact_check", self._fact_check_node)
        graph.add_node("retry_decision", self._retry_decision_node)
        
        # 添加边
        graph.set_entry_point("think")
        
        # 思考节点 → 决定调用工具还是直接总结
        graph.add_conditional_edges(
            "think",
            self._should_call_tool,
            {
                "tool_call": "tool_call",
                "summarize": "summarize",
                "fact_check": "fact_check",
                "end": END
            }
        )
        
        # 工具调用 → 重试决策（带异常处理）
        graph.add_edge("tool_call", "retry_decision")
        
        # 重试决策 → 评估或重试工具调用
        graph.add_conditional_edges(
            "retry_decision",
            self._should_retry,
            {
                "retry": "tool_call",
                "evaluate": "evaluate",
                "summarize": "summarize"
            }
        )
        
        # 评估 → 决定是否继续或总结
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "continue": "think",
                "summarize": "summarize",
                "end": END
            }
        )
        
        # 事实核查 → 总结
        graph.add_edge("fact_check", "summarize")
        
        # 总结 → 结束
        graph.add_edge("summarize", END)
        
        return graph.compile()
    
    def _think_node(self, state: AgentState) -> AgentState:
        """
        思考节点：分析用户意图，决定下一步行动
        
        :param state: 当前状态
        :return: 更新后的状态
        """
        logger.info("[LangGraphAgent] 进入思考节点")
        
        # 检查步数限制
        state["step_count"] = state.get("step_count", 0) + 1
        if state["step_count"] >= self.max_steps:
            logger.warning(f"[LangGraphAgent] 已达到最大步数限制: {self.max_steps}")
            return state
        
        messages = state.get("messages", [])
        if not messages:
            return state
        
        last_message = messages[-1]
        
        # 分析用户意图
        intent = self._analyze_intent(last_message.get("content", ""))
        
        # 判断是否需要调用工具
        need_tool = self._need_tool_call(last_message.get("content", ""), intent)
        
        if need_tool:
            # 生成工具调用
            tool_call = self._generate_tool_call(last_message.get("content", ""), intent)
            if tool_call:
                new_message = AIMessage(
                    content="",
                    tool_calls=[tool_call]
                )
                messages.append(new_message)
                state["current_task"] = f"执行工具: {tool_call.get('name', '')}"
        
        state["messages"] = messages
        return state
    
    def _analyze_intent(self, content: str) -> str:
        """
        分析用户意图
        
        :param content: 用户输入内容
        :return: 意图类型
        """
        intent_keywords = {
            "knowledge_query": ["什么是", "解释", "说明", "定义", "原理", "知识"],
            "task_request": ["帮我", "请", "需要", "制定", "规划", "创建"],
            "web_search": ["搜索", "最新", "新闻", "资讯", "现在"],
            "file_upload": ["上传", "文件", "导入"],
            "report": ["报告", "总结", "统计"],
            "chat": ["你好", "嗨", "聊天"]
        }
        
        content_lower = content.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return intent
        return "general"
    
    def _need_tool_call(self, content: str, intent: str) -> bool:
        """
        判断是否需要调用工具
        
        :param content: 用户输入内容
        :param intent: 意图类型
        :return: 是否需要调用工具
        """
        tool_intents = ["knowledge_query", "task_request", "web_search", "file_upload", "report"]
        return intent in tool_intents
    
    def _generate_tool_call(self, content: str, intent: str) -> Optional[Dict[str, Any]]:
        """
        生成工具调用
        
        :param content: 用户输入内容
        :param intent: 意图类型
        :return: 工具调用信息
        """
        intent_tool_map = {
            "knowledge_query": "rag_summarize",
            "task_request": "task_decompose",
            "web_search": "web_search",
            "report": "get_user_history"
        }
        
        tool_name = intent_tool_map.get(intent)
        
        if tool_name:
            return {
                "name": tool_name,
                "parameters": {
                    "query": content if intent in ["knowledge_query", "web_search"] else content
                }
            }
        return None
    
    def _should_call_tool(self, state: AgentState) -> str:
        """
        决定下一步：调用工具、直接总结、事实核查还是结束
        
        :param state: 当前状态
        :return: 下一步节点名称
        """
        # 检查步数限制
        if state.get("step_count", 0) >= self.max_steps:
            logger.warning(f"[LangGraphAgent] 步数超限，直接总结")
            return "summarize"
        
        messages = state.get("messages", [])
        if not messages:
            return "summarize"
        
        last_message = messages[-1]
        
        # 如果有工具调用，执行工具调用
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tool_call"
        
        # 默认直接总结
        return "summarize"
    
    def _tool_call_node(self, state: AgentState) -> AgentState:
        """
        工具调用节点：执行工具操作，带异常捕获和重试机制
        
        :param state: 当前状态
        :return: 更新后的状态
        """
        logger.info("[LangGraphAgent] 进入工具调用节点")
        
        try:
            messages = state.get("messages", [])
            if not messages:
                return state
            
            last_message = messages[-1]
            
            # 查找工具调用
            tool_call = None
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
            
            if not tool_call:
                return state
            
            tool_name = tool_call.get('name')
            tool_params = tool_call.get('parameters', {})
            
            # 查找工具
            tool = None
            for t in self.tools:
                if hasattr(t, 'name') and t.name == tool_name:
                    tool = t
                    break
            
            if not tool:
                raise ValueError(f"工具 {tool_name} 不存在")
            
            # 执行工具调用
            logger.info(f"[LangGraphAgent] 调用工具: {tool_name}, 参数: {tool_params}")
            result = tool.invoke(tool_params)
            
            # 添加工具结果消息
            tool_result_msg = ToolMessage(
                content=str(result),
                tool_call_id=tool_call.get('id', '1')
            )
            messages.append(tool_result_msg)
            
            # 重置重试计数
            state["retry_count"] = 0
            state["last_error"] = None
            
        except Exception as e:
            logger.error(f"[LangGraphAgent] 工具调用失败: {e}")
            state["retry_count"] = state.get("retry_count", 0) + 1
            state["last_error"] = str(e)
            
            # 添加错误消息
            error_msg = {
                "role": "tool",
                "content": f"工具调用失败: {str(e)}",
                "name": tool_name if 'tool_name' in locals() else "unknown"
            }
            messages.append(error_msg)
        
        state["messages"] = messages
        return state
    
    def _retry_decision_node(self, state: AgentState) -> AgentState:
        """
        重试决策节点：决定是否重试工具调用
        
        :param state: 当前状态
        :return: 更新后的状态
        """
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error")
        
        if last_error and retry_count < self.max_retries:
            logger.info(f"[LangGraphAgent] 准备重试，当前重试次数: {retry_count}/{self.max_retries}")
            # 等待重试延迟
            time.sleep(self.retry_delay)
        
        return state
    
    def _should_retry(self, state: AgentState) -> str:
        """
        判断是否需要重试
        
        :param state: 当前状态
        :return: 下一步节点名称
        """
        retry_count = state.get("retry_count", 0)
        last_error = state.get("last_error")
        
        if last_error and retry_count < self.max_retries:
            return "retry"
        elif last_error and retry_count >= self.max_retries:
            # 重试次数耗尽，直接总结
            logger.warning(f"[LangGraphAgent] 重试次数耗尽 ({self.max_retries})")
            return "summarize"
        
        return "evaluate"
    
    def _evaluate_node(self, state: AgentState) -> AgentState:
        """
        评估节点：评估工具执行结果质量
        
        :param state: 当前状态
        :return: 更新后的状态
        """
        logger.info("[LangGraphAgent] 进入评估节点")
        
        messages = state.get("messages", [])
        if not messages:
            return state
        
        # 获取工具结果
        tool_results = [m for m in messages if isinstance(m, ToolMessage) or m.get("role") == "tool"]
        
        if tool_results:
            last_result = tool_results[-1]
            
            if isinstance(last_result, ToolMessage):
                content = last_result.content
            else:
                content = last_result.get("content", "")
            
            # 简单评估：检查结果是否有效
            if content and ("success" in content.lower() or len(content) > 50):
                state["evaluation_score"] = 0.8
            else:
                state["evaluation_score"] = 0.3
                
            # 保存工具结果
            state["tool_results"].append({
                "content": content,
                "name": last_result.name if hasattr(last_result, 'name') else "unknown"
            })
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        决定是否继续执行还是总结
        
        :param state: 当前状态
        :return: 下一步节点名称
        """
        # 检查步数限制
        if state.get("step_count", 0) >= self.max_steps:
            return "end"
        
        score = state.get("evaluation_score", 0.0)
        
        # 如果分数太低，继续尝试
        if score < 0.5:
            return "continue"
        
        return "summarize"
    
    def _fact_check_node(self, state: AgentState) -> AgentState:
        """
        事实核查节点：校验答案与参考资料一致性
        
        :param state: 当前状态
        :return: 更新后的状态
        """
        logger.info("[LangGraphAgent] 进入事实核查节点")
        
        tool_results = state.get("tool_results", [])
        sources = "\n".join([r.get("content", "") for r in tool_results])
        
        messages = state.get("messages", [])
        answer = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                answer = m.content
                break
        
        if answer and sources:
            answer_keywords = set(answer.lower().split())
            source_keywords = set(sources.lower().split())
            overlap = answer_keywords.intersection(source_keywords)
            
            if len(overlap) / len(answer_keywords) > 0.5:
                logger.info("[LangGraphAgent] 事实核查通过")
            else:
                logger.warning("[LangGraphAgent] 事实核查警告：答案与来源相关性较低")
        
        return state
    
    def _summarize_node(self, state: AgentState) -> AgentState:
        """
        总结节点：生成最终回复
        
        :param state: 当前状态
        :return: 更新后的状态
        """
        logger.info("[LangGraphAgent] 进入总结节点")
        
        messages = state.get("messages", [])
        tool_results = state.get("tool_results", [])
        last_error = state.get("last_error")
        
        context = "\n".join([r.get("content", "") for r in tool_results])
        
        user_query = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                user_query = m.content
                break
        
        # 处理错误情况
        if last_error:
            final_answer = f"在处理您的请求时遇到问题：{last_error}\n\n我将基于现有信息为您提供回答：\n\n{context}" if context else f"处理失败：{last_error}"
        elif context:
            summary_prompt = f"""
基于以下信息，为用户提供一个详细的总结：

用户问题：{user_query}

参考资料：
{context}

请用自然、友好的语言总结以上信息，回答用户的问题。
            """
            
            try:
                response = chat_model.chat([HumanMessage(content=summary_prompt)])
                final_answer = response if isinstance(response, str) else str(response)
            except Exception as e:
                logger.error(f"[LangGraphAgent] 总结生成失败: {e}")
                final_answer = f"根据检索到的信息，为您总结如下：\n\n{context}"
        else:
            final_answer = self._direct_answer(user_query)
        
        messages.append({
            "role": "assistant",
            "content": final_answer
        })
        
        state["messages"] = messages
        state["final_answer"] = final_answer
        state["is_finished"] = True
        
        if tool_results and not last_error:
            evaluate_rag_pipeline(user_query, final_answer, tool_results)
        
        return state
    
    def _direct_answer_stream(self, query: str):
        """
        流式直接回答用户问题
        
        :param query: 用户问题
        :yield: 回答内容片段
        """
        try:
            response = chat_model.chat_stream([HumanMessage(content=query)])
            for chunk in response:
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"[LangGraphAgent] 流式回答失败: {e}")
            yield f"处理时遇到问题：{str(e)}"
    
    def _direct_answer(self, query: str) -> str:
        """
        直接回答用户问题（不调用工具时）
        
        :param query: 用户问题
        :return: 回答内容
        """
        try:
            response = chat_model.chat([HumanMessage(content=query)])
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"[LangGraphAgent] 直接回答失败: {e}")
            return f"处理时遇到问题：{str(e)}"
    
    async def save_state(self, session_id: str, state: AgentState):
        """
        保存工作流状态
        
        :param session_id: 会话ID
        :param state: 状态对象
        """
        if DATABASE_AVAILABLE:
            await self._save_state_to_db(session_id, state)
        else:
            self._save_state_to_file(session_id, state)
    
    async def load_state(self, session_id: str) -> Optional[AgentState]:
        """
        加载工作流状态
        
        :param session_id: 会话ID
        :return: 状态对象（如果存在）
        """
        if DATABASE_AVAILABLE:
            return await self._load_state_from_db(session_id)
        else:
            return self._load_state_from_file(session_id)
    
    async def _save_state_to_db(self, session_id: str, state: AgentState):
        """保存状态到数据库"""
        try:
            async with SessionLocal() as session:
                state_json = json.dumps(state, ensure_ascii=False)
                
                # 查找现有记录
                existing = await session.get(WorkflowState, session_id)
                
                if existing:
                    existing.state = state_json
                    existing.updated_at = func.now()
                else:
                    new_state = WorkflowState(
                        session_id=session_id,
                        state=state_json,
                        is_finished=state.get("is_finished", False)
                    )
                    session.add(new_state)
                
                await session.commit()
                logger.info(f"[LangGraphAgent] 状态已保存到数据库: {session_id}")
        except Exception as e:
            logger.error(f"[LangGraphAgent] 保存状态到数据库失败: {e}")
            # 降级到文件存储
            self._save_state_to_file(session_id, state)
    
    async def _load_state_from_db(self, session_id: str) -> Optional[AgentState]:
        """从数据库加载状态"""
        try:
            async with SessionLocal() as session:
                record = await session.get(WorkflowState, session_id)
                if record:
                    return json.loads(record.state)
            return None
        except Exception as e:
            logger.error(f"[LangGraphAgent] 从数据库加载状态失败: {e}")
            return self._load_state_from_file(session_id)
    
    def _save_state_to_file(self, session_id: str, state: AgentState):
        """保存状态到本地文件"""
        try:
            file_path = os.path.join(self.persist_dir, f"{session_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.info(f"[LangGraphAgent] 状态已保存到文件: {file_path}")
        except Exception as e:
            logger.error(f"[LangGraphAgent] 保存状态到文件失败: {e}")
    
    def _load_state_from_file(self, session_id: str) -> Optional[AgentState]:
        """从本地文件加载状态"""
        try:
            file_path = os.path.join(self.persist_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"[LangGraphAgent] 从文件加载状态失败: {e}")
            return None
    
    def run(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """
        运行工作流（非流式）
        
        :param user_input: 用户输入
        :param session_id: 会话ID（用于状态持久化）
        :return: 执行结果
        """
        logger.info(f"[LangGraphAgent] 开始处理用户输入: {user_input}")
        
        initial_state = AgentState(
            messages=[
                HumanMessage(content=user_input)
            ],
            task_list=[],
            tool_results=[],
            is_finished=False,
            step_count=0,
            retry_count=0,
            session_id=session_id
        )
        
        try:
            result = self.workflow.invoke(initial_state)
            
            # 保存最终状态
            if session_id:
                import asyncio
                asyncio.run(self.save_state(session_id, result))
            
            return result
        except Exception as e:
            logger.error(f"[LangGraphAgent] 工作流执行失败: {e}", exc_info=True)
            
            # 保存错误状态
            if session_id:
                error_state = AgentState(
                    messages=[HumanMessage(content=user_input)],
                    is_finished=True,
                    final_answer=f"处理过程中出现错误：{str(e)}",
                    last_error=str(e),
                    session_id=session_id
                )
                import asyncio
                asyncio.run(self.save_state(session_id, error_state))
            
            return {
                "final_answer": f"处理过程中出现错误：{str(e)}",
                "is_finished": True,
                "error": str(e)
            }
    
    def run_stream(self, user_input: str):
        """
        运行工作流（流式输出）
        
        :param user_input: 用户输入
        :yield: 回答内容片段
        """
        logger.info(f"[LangGraphAgent] 开始流式处理用户输入: {user_input}")
        
        intent = self._analyze_intent(user_input)
        need_tool = self._need_tool_call(user_input, intent)
        
        if not need_tool:
            logger.info(f"[LangGraphAgent] 简单问题，直接流式回答")
            for chunk in self._direct_answer_stream(user_input):
                yield chunk
            return
        
        try:
            initial_state = AgentState(
                messages=[
                    HumanMessage(content=user_input)
                ],
                task_list=[],
                tool_results=[],
                is_finished=False,
                step_count=0,
                retry_count=0
            )
            
            result = self.workflow.invoke(initial_state)
            tool_results = result.get("tool_results", [])
            last_error = result.get("last_error")
            
            context = "\n".join([r.get("content", "") for r in tool_results])
            
            if last_error:
                yield f"在处理您的请求时遇到问题：{last_error}\n\n"
            
            if context:
                summary_prompt = f"""
基于以下信息，为用户提供一个详细的总结：

用户问题：{user_input}

参考资料：
{context}

请用自然、友好的语言总结以上信息，回答用户的问题。
                """
                
                for chunk in chat_model.chat_stream([HumanMessage(content=summary_prompt)]):
                    if chunk:
                        yield chunk
            else:
                for chunk in self._direct_answer_stream(user_input):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"[LangGraphAgent] 流式工作流执行失败: {e}")
            yield f"处理过程中出现错误：{str(e)}"


# 创建全局实例
langgraph_agent = LangGraphAgent()


def create_langgraph_agent(tools: List[BaseTool] = None, max_steps: int = 10, 
                           max_retries: int = 3) -> LangGraphAgent:
    """
    创建 LangGraph Agent 实例
    
    :param tools: 工具列表
    :param max_steps: 最大步数限制
    :param max_retries: 最大重试次数
    :return: LangGraphAgent 实例
    """
    return LangGraphAgent(tools=tools, max_steps=max_steps, max_retries=max_retries)


def run_workflow(user_input: str, tools: List[BaseTool] = None, session_id: str = None) -> Dict[str, Any]:
    """
    便捷函数：运行工作流（非流式）
    
    :param user_input: 用户输入
    :param tools: 工具列表
    :param session_id: 会话ID
    :return: 执行结果
    """
    agent = LangGraphAgent(tools=tools)
    return agent.run(user_input, session_id=session_id)


def run_workflow_stream(user_input: str, tools: List[BaseTool] = None):
    """
    便捷函数：运行工作流（流式）
    
    :param user_input: 用户输入
    :param tools: 工具列表
    :yield: 回答内容片段
    """
    agent = LangGraphAgent(tools=tools)
    return agent.run_stream(user_input)
