import time
from typing import Any, List, Dict, Optional, Tuple

from langchain.agents import create_agent
from agent.session import AgentSession
from agent.tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fill_context_report,
    get_user_history,
    task_decompose,
    evaluate_result,
    fact_check,
    web_search,
    fetch_webpage
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.modules import task_decomposer, path_planner, self_evaluator, fact_checker, dynamic_adjuster
from model.factory import chat_model
from utils.config_handler import agent_cfg
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt


class ReactAgent:
    def __init__(self):
        self.tools = [
            # 核心工具
            rag_summarize,
            # 网络搜索工具
            web_search,
            fetch_webpage,
            # 通用工具
            get_weather,
            get_user_location,
            get_user_id,
            get_current_month,
            # 报告工具
            fill_context_report,
            get_user_history,
            # 高级能力工具
            task_decompose,
            evaluate_result,
            fact_check,
        ]
        
        # Disable tools for vision models to prevent streaming errors
        if 'vl' in chat_model.model.lower() or 'vision' in chat_model.model.lower():
            self.tools = []
            logger.warning("Vision model detected, disabling tools to prevent streaming errors")
        
        try:
            self.agent = create_agent(
                model=chat_model,
                system_prompt=load_system_prompt(),
                tools=self.tools,
                middleware=[log_before_model, monitor_tool, report_prompt_switch],
            )
            logger.info(f"Agent created with {len(self.tools)} tools")
        except NotImplementedError:
            logger.warning("Model does not support tools, creating agent without tools")
            self.agent = create_agent(
                model=chat_model,
                system_prompt=load_system_prompt(),
                tools=[],
                middleware=[log_before_model, monitor_tool, report_prompt_switch],
            )
        self.session: AgentSession | None = None
        self.retry_max_attempts = int(agent_cfg.get("retry_max_attempts", 3))
        self.retry_backoff_seconds = float(agent_cfg.get("retry_backoff_seconds", 2))
        
        # 高级能力模块
        self.task_decomposer = task_decomposer
        self.path_planner = path_planner
        self.self_evaluator = self_evaluator
        self.fact_checker = fact_checker
        self.dynamic_adjuster = dynamic_adjuster
        
        # 启用/禁用高级能力
        enabled = agent_cfg.get("enable_advanced_features", True)
        if isinstance(enabled, str):
            self.enable_advanced_features = enabled.lower() == "true"
        else:
            self.enable_advanced_features = bool(enabled)
        logger.info(f"[ReactAgent] 高级能力模块已{'启用' if self.enable_advanced_features else '禁用'}")

    def _make_session(self, query: str) -> None:
        self.session = AgentSession()
        self.session.log(
            "query_received",
            {
                "query": query,
                "tool_count": len(self.tools),
                "model_source": agent_cfg.get("model_source", "cloud"),
            },
        )

    def _stream_with_retries(self, messages: list[dict[str, str]], context: dict[str, Any]):
        attempt = 0
        while True:
            attempt += 1
            yielded_any = False
            try:
                for chunk in self.agent.stream({"messages": messages}, config={"configurable": context}, stream_mode="messages"):
                    if chunk is None:
                        continue
                    yielded_any = True
                    yield chunk
                return
            except Exception as exc:
                import traceback
                error_details = f"{type(exc).__name__}: {exc}"
                error_trace = traceback.format_exc()
                logger.error(
                    f"[ReactAgent] 模型流式调用失败，第 {attempt} 次重试，错误：{error_details}"
                )
                logger.error(f"[ReactAgent] 错误堆栈：\n{error_trace}")
                self.session.log(
                    "retry",
                    {
                        "attempt": attempt,
                        "error": str(exc),
                        "max_attempts": self.retry_max_attempts,
                        "yielded_any": yielded_any,
                    },
                )
                if yielded_any or attempt >= self.retry_max_attempts:
                    self.session.log(
                        "fatal_error",
                        {"attempt": attempt, "error": str(exc), "yielded_any": yielded_any},
                    )
                    raise
                backoff = self.retry_backoff_seconds * attempt
                time.sleep(backoff)
                continue

    def execute_stream(self, query: str, max_iterations: int = 3):
        self._make_session(query)
        messages = [{"role": "user", "content": query}]
        context = {"report": False, "session_id": self.session.session_id}

        self.session.log("observe", {"messages": messages, "context": context})

        last_yielded = ""
        for chunk in self._stream_with_retries(messages, context):
            # langgraph agent.stream() 返回的是 (message, context) tuple
            if isinstance(chunk, (list, tuple)) and len(chunk) >= 1:
                latest_message = chunk[0]
            elif isinstance(chunk, dict):
                messages_in_chunk = chunk.get("messages")
                if not messages_in_chunk:
                    continue
                latest_message = messages_in_chunk[-1]
            else:
                continue

            # 跳过 ToolMessage（工具返回结果，不需要输出）
            from langchain_core.messages import ToolMessage
            if isinstance(latest_message, ToolMessage):
                continue

            if hasattr(latest_message, "content") and latest_message.content is not None:
                content = latest_message.content.strip()
                if content != last_yielded:
                    if content.startswith(last_yielded):
                        delta = content[len(last_yielded) :]
                    else:
                        delta = content
                    last_yielded = content
                    self.session.log("stream_chunk", {"chunk": delta})
                    yield delta

    def stream_to_text(self, query: str, max_iterations: int = 3) -> str:
        return "".join(self.execute_stream(query, max_iterations=max_iterations))
    
    def _analyze_complexity(self, query: str) -> bool:
        """分析查询是否为复杂任务"""
        complexity_indicators = [
            "帮我", "请帮我", "如何", "方案", "优化", "分析", "总结",
            "步骤", "流程", "计划", "规划", "设计", "制定"
        ]
        return any(indicator in query for indicator in complexity_indicators)
    
    def execute_with_planning(self, query: str) -> str:
        """
        使用高级规划能力执行复杂任务
        
        :param query: 用户查询
        :return: 最终答案
        """
        if not self.enable_advanced_features:
            logger.info("[ReactAgent] 高级能力未启用，使用普通执行模式")
            return self.stream_to_text(query)
        
        if not self._analyze_complexity(query):
            logger.info("[ReactAgent] 检测到简单查询，使用普通执行模式")
            return self.stream_to_text(query)
        
        logger.info(f"[ReactAgent] 检测到复杂任务，启动高级规划模式: {query}")
        
        # 步骤1: 任务拆解
        logger.info("[ReactAgent] 步骤1: 任务拆解")
        tasks = self.task_decomposer.decompose(query)
        logger.info(f"[ReactAgent] 拆解出 {len(tasks)} 个子任务")
        
        if not tasks:
            logger.warning("[ReactAgent] 任务拆解失败，回退到普通模式")
            return self.stream_to_text(query)
        
        # 步骤2: 路径规划
        logger.info("[ReactAgent] 步骤2: 路径规划")
        planned_tasks = self.path_planner.plan(tasks)
        
        # 步骤3: 执行计划
        logger.info("[ReactAgent] 步骤3: 执行计划")
        execution_results = []
        
        for task in planned_tasks:
            result = self._execute_task(task, execution_results)
            execution_results.append({"task": task, "result": result})
            
            # 步骤4: 自我评估
            logger.info(f"[ReactAgent] 步骤4: 自我评估任务 {task.get('id')}")
            quality, feedback, need_retry = self.self_evaluator.evaluate(task, result)
            
            if need_retry:
                logger.info(f"[ReactAgent] 任务 {task.get('id')} 质量不足，尝试调整策略")
                adjustment = self.dynamic_adjuster.adjust(task, result=result, quality_score=quality)
                
                if adjustment["action"] == "retry":
                    result = self._execute_task(task, execution_results)
                elif adjustment["action"] == "skip":
                    logger.info(f"[ReactAgent] 跳过任务 {task.get('id')}")
        
        # 步骤5: 整合结果并进行事实核查
        logger.info("[ReactAgent] 步骤5: 整合结果并事实核查")
        final_answer = self._synthesize_answer(query, execution_results)
        
        # 事实核查
        sources = self._collect_sources(execution_results)
        confidence, inconsistencies, suggestion = self.fact_checker.check(final_answer, sources)
        
        if confidence < 0.7:
            logger.warning(f"[ReactAgent] 事实核查失败，置信度: {confidence}")
            logger.warning(f"[ReactAgent] 不一致项: {inconsistencies}")
            final_answer = f"{final_answer}\n\n⚠️ 注意：此答案部分内容与参考资料存在差异。\n建议：{suggestion}"
        
        return final_answer
    
    def _execute_task(self, task: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> str:
        """执行单个子任务"""
        tool = task.get("tool")
        
        if not tool:
            # 不需要工具的任务，直接生成结果
            description = task.get("description", "")
            return f"任务完成：{description}"
        
        # 查找工具并执行
        tool_func = next((t for t in self.tools if t.name == tool), None)
        
        if not tool_func:
            logger.warning(f"[ReactAgent] 未找到工具: {tool}")
            return f"未找到工具 {tool}"
        
        try:
            # 从之前的结果中提取参数
            params = task.get("params", {})
            if not params:
                params = {"query": task.get("title", "")}
            
            result = tool_func.invoke(params)
            return str(result)
        except Exception as e:
            logger.error(f"[ReactAgent] 工具执行失败: {e}")
            return f"工具执行失败: {str(e)}"
    
    def _synthesize_answer(self, query: str, execution_results: List[Dict[str, Any]]) -> str:
        """整合所有任务结果生成最终答案"""
        prompt = f"""请根据以下执行结果，为用户问题生成一个完整的回答：

用户问题：{query}

执行结果：
"""
        for i, item in enumerate(execution_results, 1):
            task = item["task"]
            result = item["result"]
            prompt += f"{i}. [{task.get('title')}] {result}\n\n"
        
        prompt += "\n请生成一个连贯、专业的最终回答："
        
        try:
            response = chat_model.invoke(prompt)
            return getattr(response, 'content', str(response))
        except Exception as e:
            logger.error(f"[ReactAgent] 答案合成失败: {e}")
            return "\n".join([str(r["result"]) for r in execution_results])
    
    def _collect_sources(self, execution_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """收集参考资料"""
        sources = []
        for item in execution_results:
            result = item["result"]
            if result:
                sources.append({
                    "page_content": str(result),
                    "metadata": {"source": item["task"].get("title", "unknown")}
                })
        return sources


if __name__ == "__main__":
    react_agent = ReactAgent()
    # 测试高级规划模式
    result = react_agent.execute_with_planning("帮我制定一个AI行业研究方案")
    print("\n" + "="*50)
    print("最终答案：")
    print(result)