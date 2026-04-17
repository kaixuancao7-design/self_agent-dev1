"""
LangGraph 工作流架构

基于 LangGraph 重构的 Agent 工作流，包含：
- 思考节点：分析用户意图，决定下一步行动
- 工具调用节点：执行工具操作
- 总结节点：生成最终回复
- 评估节点：评估执行结果质量
- 事实核查节点：校验答案与参考资料一致性

工作流设计遵循 ReAct 模式：思考 → 行动 → 观察 → 总结
"""

import json
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from utils.logger_handler import logger
from model.factory import chat_model
from rag.ragas_evaluator import evaluate_rag_pipeline


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
    """
    messages: List[Dict[str, Any]]
    current_task: Optional[str] = None
    task_list: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    is_finished: bool = False
    final_answer: Optional[str] = None
    evaluation_score: Optional[float] = None


class LangGraphAgent:
    """基于 LangGraph 的 Agent 工作流"""
    
    def __init__(self, tools: List[BaseTool] = None):
        """
        初始化 LangGraph Agent
        
        :param tools: 可用工具列表
        """
        self.tools = tools or []
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        graph = StateGraph(AgentState)
        
        # 添加节点
        graph.add_node("think", self._think_node)
        graph.add_node("tool_call", ToolNode(self.tools))
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("fact_check", self._fact_check_node)
        
        # 添加边
        graph.set_entry_point("think")
        
        # 思考节点 → 决定调用工具还是直接总结
        graph.add_conditional_edges(
            "think",
            self._should_call_tool,
            {
                "tool_call": "tool_call",
                "summarize": "summarize",
                "fact_check": "fact_check"
            }
        )
        
        # 工具调用 → 评估结果
        graph.add_edge("tool_call", "evaluate")
        
        # 评估 → 决定是否继续或总结
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "continue": "think",
                "summarize": "summarize"
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
        
        # 获取最新消息
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
        # 根据意图选择合适的工具
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
        决定下一步：调用工具、直接总结还是进行事实核查
        
        :param state: 当前状态
        :return: 下一步节点名称
        """
        messages = state.get("messages", [])
        if not messages:
            return "summarize"
        
        last_message = messages[-1]
        
        # 如果有工具调用，执行工具调用
        if "tool_calls" in last_message and last_message["tool_calls"]:
            return "tool_call"
        
        # 如果有工具结果，进行评估后决定
        if last_message.get("role") == "tool":
            return "evaluate"
        
        # 默认直接总结
        return "summarize"
    
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
        tool_results = [m for m in messages if m.get("role") == "tool"]
        
        if tool_results:
            last_result = tool_results[-1]
            content = last_result.get("content", "")
            
            # 简单评估：检查结果是否有效
            if content and ("success" in content.lower() or len(content) > 50):
                state["evaluation_score"] = 0.8  # 高分
            else:
                state["evaluation_score"] = 0.3  # 低分
                
            # 保存工具结果
            state["tool_results"].append(last_result)
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        决定是否继续执行还是总结
        
        :param state: 当前状态
        :return: 下一步节点名称
        """
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
        
        # 提取工具结果作为参考来源
        tool_results = state.get("tool_results", [])
        sources = "\n".join([r.get("content", "") for r in tool_results])
        
        # 提取待核查的答案（如果有的话）
        messages = state.get("messages", [])
        answer = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                answer = m.get("content", "")
                break
        
        if answer and sources:
            # 简单事实核查：检查答案中的关键词是否在来源中出现
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
        
        # 收集工具结果
        context = "\n".join([r.get("content", "") for r in tool_results])
        
        # 获取用户原始问题
        user_query = ""
        for m in messages:
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break
        
        # 如果有工具结果，基于结果生成回复
        if context:
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
            # 没有工具结果，直接回答
            final_answer = self._direct_answer(user_query)
        
        # 添加总结消息
        messages.append({
            "role": "assistant",
            "content": final_answer
        })
        
        state["messages"] = messages
        state["final_answer"] = final_answer
        state["is_finished"] = True
        
        # 评估整体效果
        if tool_results:
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
            yield "我来帮您解答这个问题。"
    
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
            return "我来帮您解答这个问题。"
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """
        运行工作流（非流式）
        
        :param user_input: 用户输入
        :return: 执行结果
        """
        logger.info(f"[LangGraphAgent] 开始处理用户输入: {user_input}")
        
        # 初始化状态
        initial_state = AgentState(
            messages=[
                HumanMessage(content=user_input)
            ],
            task_list=[],
            tool_results=[],
            is_finished=False
        )
        
        # 执行工作流
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"[LangGraphAgent] 工作流执行失败: {e}")
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
        
        # 首先判断是否需要调用工具
        intent = self._analyze_intent(user_input)
        need_tool = self._need_tool_call(user_input, intent)
        
        if not need_tool:
            # 简单问题，直接流式回答
            logger.info(f"[LangGraphAgent] 简单问题，直接流式回答")
            for chunk in self._direct_answer_stream(user_input):
                yield chunk
            return
        
        # 需要调用工具的复杂问题
        # 先执行工具调用（同步），然后流式生成总结
        logger.info(f"[LangGraphAgent] 复杂问题，先执行工具调用")
        
        try:
            # 执行工作流获取工具结果
            initial_state = AgentState(
                messages=[
                    HumanMessage(content=user_input)
                ],
                task_list=[],
                tool_results=[],
                is_finished=False
            )
            
            result = self.workflow.invoke(initial_state)
            tool_results = result.get("tool_results", [])
            
            # 收集工具结果
            context = "\n".join([r.get("content", "") for r in tool_results])
            
            if context:
                summary_prompt = f"""
基于以下信息，为用户提供一个详细的总结：

用户问题：{user_input}

参考资料：
{context}

请用自然、友好的语言总结以上信息，回答用户的问题。
                """
                
                # 流式生成总结
                for chunk in chat_model.chat_stream([HumanMessage(content=summary_prompt)]):
                    if chunk:
                        yield chunk
            else:
                # 没有工具结果，直接流式回答
                for chunk in self._direct_answer_stream(user_input):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"[LangGraphAgent] 流式工作流执行失败: {e}")
            yield f"处理过程中出现错误：{str(e)}"


# 创建全局实例
langgraph_agent = LangGraphAgent()


def create_langgraph_agent(tools: List[BaseTool] = None) -> LangGraphAgent:
    """
    创建 LangGraph Agent 实例
    
    :param tools: 工具列表
    :return: LangGraphAgent 实例
    """
    return LangGraphAgent(tools=tools)


def run_workflow(user_input: str, tools: List[BaseTool] = None) -> Dict[str, Any]:
    """
    便捷函数：运行工作流（非流式）
    
    :param user_input: 用户输入
    :param tools: 工具列表
    :return: 执行结果
    """
    agent = LangGraphAgent(tools=tools)
    return agent.run(user_input)


def run_workflow_stream(user_input: str, tools: List[BaseTool] = None):
    """
    便捷函数：运行工作流（流式）
    
    :param user_input: 用户输入
    :param tools: 工具列表
    :yield: 回答内容片段
    """
    agent = LangGraphAgent(tools=tools)
    return agent.run_stream(user_input)
