import time
import difflib
from typing import Any, List, Dict, Optional, Tuple

from langchain.agents import create_agent
from agent.session import AgentSession
from agent.skills import skill_loader
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools import (
    fetch_webpage,
    get_user_location,
    get_user_id,
    get_current_month,
    fill_context_report,
    get_user_history,
    evaluate_result,
    get_knowledge_base_stats,
    list_databases,
    create_database,
    delete_database,
    switch_database,
    list_uploaded_files,
    remove_file_from_knowledge_base,
    reparse_file_in_knowledge_base
)

# Skills 框架已包含的工具：rag_summarize, web_search, get_weather, task_decompose, fact_check, kb_management
from agent.modules import (
    task_decomposer, path_planner, self_evaluator, fact_checker, dynamic_adjuster,
    memory_manager, intent_recognizer
)
from model.factory import chat_model
from utils.config_handler import agent_cfg
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt


class ReactAgent:
    def __init__(self):
        # 使用 Skills 框架加载技能
        skill_tools = skill_loader.get_tool_list()
        
        self.tools = [
            # 从 Skills 加载的工具
            *skill_tools,
            # 额外的工具
            fetch_webpage,
            get_user_location,
            get_user_id,
            get_current_month,
            fill_context_report,
            get_user_history,
            evaluate_result,
            # 知识库管理工具
            get_knowledge_base_stats,
            list_databases,
            create_database,
            delete_database,
            switch_database,
            list_uploaded_files,
            remove_file_from_knowledge_base,
            reparse_file_in_knowledge_base,
        ]
        
        # Disable tools for vision models to prevent streaming errors
        model_name = ""
        if hasattr(chat_model, 'model'):
            model_name = chat_model.model
        elif hasattr(chat_model, '_model_name'):
            model_name = chat_model._model_name
        elif hasattr(chat_model, 'model_name'):
            model_name = chat_model.model_name
        
        if model_name and ('vl' in model_name.lower() or 'vision' in model_name.lower()):
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
        
        # 记忆管理模块
        self.memory_manager = memory_manager
        
        # 意图识别模块
        self.intent_recognizer = intent_recognizer
        
        # 启用/禁用高级能力
        enabled = agent_cfg.get("enable_advanced_features", True)
        if isinstance(enabled, str):
            self.enable_advanced_features = enabled.lower() == "true"
        else:
            self.enable_advanced_features = bool(enabled)
        logger.info(f"[ReactAgent] 高级能力模块已{'启用' if self.enable_advanced_features else '禁用'}")
        
        # 启用/禁用记忆管理
        memory_enabled = agent_cfg.get("enable_memory", True)
        if isinstance(memory_enabled, str):
            self.enable_memory = memory_enabled.lower() == "true"
        else:
            self.enable_memory = bool(memory_enabled)
        logger.info(f"[ReactAgent] 记忆管理模块已{'启用' if self.enable_memory else '禁用'}")

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

    def _execute_tool_call(self, tool_call_content: str) -> str:
        """执行工具调用并返回结果"""
        import json
        
        try:
            # 提取JSON内容
            json_start = tool_call_content.find('{')
            json_end = tool_call_content.rfind('}')
            
            if json_start == -1 or json_end == -1:
                logger.warning(f"[ReactAgent] 无法解析工具调用JSON: {tool_call_content[:50]}")
                return ""
            
            json_content = tool_call_content[json_start:json_end+1]
            
            # 解析JSON
            tool_call = json.loads(json_content)
            
            # 提取工具名称和参数（支持多种格式）
            tool_name = None
            params = {}
            
            # 格式1: {"tool_name": "...", "params": {...}}
            if "tool_name" in tool_call:
                tool_name = tool_call["tool_name"]
                params = tool_call.get("params", {})
            # 格式2: {"tools": [...]}
            elif "tools" in tool_call and isinstance(tool_call["tools"], list):
                if tool_call["tools"]:
                    tool_name = tool_call["tools"][0].get("name")
                    params = tool_call["tools"][0].get("arguments", {})
            # 格式3: {"tool": "...", "parameters": {...}}
            elif "tool" in tool_call:
                tool_name = tool_call["tool"]
                params = tool_call.get("parameters", tool_call.get("params", {}))
            # 格式4: {"skill": "...", "params": {...}}
            elif "skill" in tool_call:
                tool_name = tool_call["skill"]
                params = tool_call.get("params", {})
            # 格式5: {"search": {...}} 或其他工具直接作为key
            else:
                for key, value in tool_call.items():
                    if isinstance(value, dict):
                        tool_name = key
                        params = value
                        break
            
            if not tool_name:
                logger.warning(f"[ReactAgent] 无法识别工具名称: {tool_call_content[:50]}")
                return ""
            
            # 查找工具并执行
            tool_func = next((t for t in self.tools if getattr(t, 'name', None) == tool_name), None)
            
            if not tool_func:
                # 获取所有可用工具名称
                available_tools = [getattr(t, 'name', 'unknown') for t in self.tools]
                logger.warning(f"[ReactAgent] 未找到工具: {tool_name}，可用工具: {available_tools}")
                
                # 尝试模糊匹配
                if available_tools:
                    matches = difflib.get_close_matches(tool_name, available_tools, n=1, cutoff=0.5)
                    if matches:
                        suggested_tool = matches[0]
                        logger.info(f"[ReactAgent] 发现相似工具: {suggested_tool}，尝试使用")
                        tool_func = next((t for t in self.tools if getattr(t, 'name', None) == suggested_tool), None)
                
                if not tool_func:
                    return f"未找到工具: '{tool_name}'。可用工具列表: {', '.join(available_tools)}"
            
            logger.info(f"[ReactAgent] 执行工具调用: {tool_name}, 参数: {params}")
            
            # 执行工具
            result = tool_func.invoke(params)
            
            # 处理结果
            result_str = str(result)
            logger.info(f"[ReactAgent] 工具执行结果: {result_str[:100]}...")
            
            return result_str
            
        except json.JSONDecodeError as e:
            logger.error(f"[ReactAgent] 解析工具调用JSON失败: {e}")
            return ""
        except Exception as e:
            logger.error(f"[ReactAgent] 执行工具调用失败: {e}", exc_info=True)
            return f"工具执行失败: {str(e)}"

    def execute_stream(self, query: str, max_iterations: int = 3):
        self._make_session(query)
        
        # 意图识别
        intent_result = self.intent_recognizer.recognize(query)
        logger.info(f"[ReactAgent] 意图识别结果: {intent_result['intent']} (置信度: {intent_result['confidence']:.2f})")
        self.session.log("intent_recognition", intent_result)
        
        # 获取记忆上下文
        if self.enable_memory:
            memory_context = self.memory_manager.retrieve_relevant_memory(query)
            if memory_context:
                query_with_context = f"【记忆参考】\n{memory_context}\n\n【当前问题】\n{query}"
            else:
                query_with_context = query
            
            # 添加用户消息到短期记忆
            self.memory_manager.add_conversation("user", query)
        else:
            query_with_context = query
        
        messages = [{"role": "user", "content": query_with_context}]
        context = {
            "report": False, 
            "session_id": self.session.session_id,
            "intent": intent_result["intent"],
            "implicit_needs": intent_result["implicit_needs"],
            "suggested_actions": intent_result["suggested_actions"]
        }

        self.session.log("observe", {"messages": messages, "context": context})

        # 缓冲所有流式输出内容，直到结束后再统一处理
        # 这样可以正确处理工具调用标记出现在最后chunk的情况
        full_content = ""
        
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
                # 记录工具调用结果到日志（后台）
                logger.debug(f"[ReactAgent] 工具返回结果: {latest_message.content[:100]}...")
                continue

            if hasattr(latest_message, "content") and latest_message.content is not None:
                full_content += latest_message.content

        # 流式输出结束，现在统一处理完整内容
        content = full_content.strip()
        
        # 定义思考过程标记
        thinking_prefixes = [
            "思考:", "Thought:", "【思考】", "思考过程：", "thinking:", 
            "思考过程:", "**思考**", "思考：", "【思考过程】", "###思考过程", "##思考过程",
            "思考步骤：", "分析过程：", "推理过程：", "思考分析："
        ]
        
        # 检查是否包含思考过程标记
        has_thinking = any(prefix in content for prefix in thinking_prefixes)
        if not has_thinking and "<思考过程>" in content:
            has_thinking = True
        
        # 处理工具调用内容（优先于思考过程处理）
        # 工具调用格式可能是：
        # 1. ```json{"tools":[...]}```
        # 2. {"tools"...} 或 {"tool_name"...}
        # 3. json{"skill"...} 或 json{"tool"...}
        # 4. 【工具调用】或 工具调用 标签格式
        tool_call_patterns = ['```json', '{"tools"', '{"tool_name"', 'json{"skill"', 'json{"tool"', '【工具调用】', '工具调用']
        tool_start = -1
        tool_pattern = ''
        
        for pattern in tool_call_patterns:
            pos = content.find(pattern)
            if pos != -1 and (tool_start == -1 or pos < tool_start):
                tool_start = pos
                tool_pattern = pattern
        
        # 定义思考过程标签（带方括号和不带方括号）
        thinking_tags = ["【思考过程】", "思考过程"]
        
        if tool_start != -1:
            # 工具调用之前的内容是正式输出
            formal_output = content[:tool_start].strip()
            # 记录工具调用到日志（后台）
            tool_call_content = content[tool_start:].strip()
            logger.info(f"[ReactAgent] 【后台工具调用】{tool_call_content[:100]}...")
            self.session.log("tool_call", {"content": tool_call_content})
            
            # 如果正式输出中包含思考过程标签，提取标签之前的内容
            thinking_tag_start = -1
            thinking_tag_found = ""
            for tag in thinking_tags:
                pos = formal_output.find(tag)
                if pos != -1 and (thinking_tag_start == -1 or pos < thinking_tag_start):
                    thinking_tag_start = pos
                    thinking_tag_found = tag
            
            if thinking_tag_start != -1:
                formal_output_before_thinking = formal_output[:thinking_tag_start].strip()
                
                # 如果思考标签之前有内容，使用它作为正式输出
                if formal_output_before_thinking:
                    formal_output = formal_output_before_thinking
                else:
                    # 如果思考标签之前没有内容，生成友好提示
                    thinking_end = -1
                    for tag in tool_call_patterns:
                        pos = content.find(tag)
                        if pos != -1 and (thinking_end == -1 or pos < thinking_end):
                            thinking_end = pos
                    
                    if thinking_end != -1:
                        thinking_content = content[thinking_tag_start + len(thinking_tag_found):thinking_end].strip()
                        keywords = []
                        if "搜索" in thinking_content:
                            keywords.append("搜索")
                        if "最新" in thinking_content:
                            keywords.append("最新")
                        if "AI" in thinking_content or "AIAgent" in thinking_content or "智能体" in thinking_content:
                            keywords.append("AIAgent")
                        if keywords:
                            formal_output = f"我正在为您{'、'.join(keywords)}相关信息，请稍候..."
                        else:
                            formal_output = "我正在为您处理请求，请稍候..."
                    else:
                        formal_output = "我正在为您处理请求，请稍候..."
            
            # 如果正式输出为空，生成友好提示
            if not formal_output:
                thinking_start = -1
                thinking_tag_found = ""
                for tag in thinking_tags:
                    pos = content.find(tag)
                    if pos != -1 and (thinking_start == -1 or pos < thinking_start):
                        thinking_start = pos
                        thinking_tag_found = tag
                
                if thinking_start != -1:
                    thinking_end = -1
                    for tag in tool_call_patterns:
                        pos = content.find(tag)
                        if pos != -1 and (thinking_end == -1 or pos < thinking_end):
                            thinking_end = pos
                    
                    if thinking_end != -1:
                        thinking_content = content[thinking_start + len(thinking_tag_found):thinking_end].strip()
                        keywords = []
                        if "搜索" in thinking_content:
                            keywords.append("搜索")
                        if "最新" in thinking_content:
                            keywords.append("最新")
                        if keywords:
                            formal_output = f"我正在为您{'、'.join(keywords)}相关信息，请稍候..."
                        else:
                            formal_output = "我正在为您处理请求，请稍候..."
                    else:
                        formal_output = "我正在为您处理请求，请稍候..."
                else:
                    formal_output = "我正在为您搜索相关信息，请稍候..."
            
            # 执行工具调用并获取结果
            tool_result = self._execute_tool_call(tool_call_content)
            
            # 如果有工具执行结果，整合到正式输出中
            if tool_result:
                formal_output = f"{formal_output}\n\n{tool_result}"
            
            content = formal_output
    
        # 处理思考过程（仅在没有工具调用时）
        if has_thinking and tool_start == -1:
            thinking_content = ""
            formal_output = ""
            original_content = content
            processed = False
            
            # 先检查是否为<思考过程>标签格式
            if "<思考过程>" in content and "</思考过程>" in content:
                start_tag = "<思考过程>"
                end_tag = "</思考过程>"
                start_pos = content.find(start_tag)
                end_pos = content.find(end_tag)
                
                if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                    thinking_content = content[start_pos + len(start_tag):end_pos].strip()
                    formal_output = content[:start_pos].strip() + content[end_pos + len(end_tag):].strip()
                    processed = True
            
            # 如果不是标签格式，检查普通思考前缀格式
            if not processed:
                first_idx = len(content)
                found_prefix = ""
                for prefix in thinking_prefixes:
                    if prefix in content:
                        idx = content.find(prefix)
                        if idx < first_idx:
                            first_idx = idx
                            found_prefix = prefix
                
                if found_prefix:
                    prefix_before_content = content[:first_idx].strip()
                    thinking_content = content[first_idx:]
                    
                    separators = [
                        '\n--\n', '\n---\n', '--\n', '---\n', '\n## ', 
                        '\n回答：', '\n回复：', '\n总结：', '\n结论：',
                        '\n**回答**', '\n**回复**', '\n### '
                    ]
                    separator_pos = -1
                    used_separator = ''
                    
                    for sep in separators:
                        pos = thinking_content.find(sep)
                        if pos != -1 and (separator_pos == -1 or pos < separator_pos):
                            separator_pos = pos
                            used_separator = sep
                    
                    if separator_pos != -1:
                        formal_output = prefix_before_content + '\n' + thinking_content[separator_pos + len(used_separator):].strip()
                        thinking_content = thinking_content[:separator_pos].strip()
                    else:
                        json_start = thinking_content.find('{"')
                        if json_start != -1:
                            thinking_content = thinking_content[:json_start].strip()
                            formal_output = prefix_before_content
                        else:
                            tool_call_markers = ["调用工具", "调用", "使用工具", "使用", "execute", "exec", "开始搜索", "开始调用"]
                            has_tool_call_marker = any(marker in thinking_content for marker in tool_call_markers)
                            
                            if has_tool_call_marker:
                                tool_start_pos = len(thinking_content)
                                for marker in tool_call_markers:
                                    pos = thinking_content.find(marker)
                                    if pos != -1 and pos < tool_start_pos:
                                        tool_start_pos = pos
                                thinking_content = thinking_content[:tool_start_pos].strip()
                                formal_output = prefix_before_content
                            else:
                                formal_output = original_content
            
            # 记录思考过程到日志（后台）
            if thinking_content:
                if "未发现合适的技能" in thinking_content or "未找到" in thinking_content or "找不到" in thinking_content:
                    logger.warning(f"[ReactAgent] 【后台标注】未发现合适的技能/工具: {thinking_content[:100]}...")
                    self.session.log("skill_not_found", {"thinking": thinking_content, "query": query})
                else:
                    logger.info(f"[ReactAgent] 【后台思考】{thinking_content[:100]}...")
                self.session.log("thinking", {"content": thinking_content})
            
            if formal_output:
                content = formal_output
        
        # 正式输出内容
        response_content = content
        
        # 确保至少 yield 一个内容（防止生成器返回 None）
        if content:
            self.session.log("stream_chunk", {"chunk": content})
            yield content
        else:
            # 如果没有内容，yield 一个友好提示
            default_response = "我已经收到您的请求，正在处理中..."
            self.session.log("stream_chunk", {"chunk": default_response})
            yield default_response

        # 将响应添加到短期记忆
        if self.enable_memory and response_content:
            self.memory_manager.add_conversation("assistant", response_content)

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
            # 获取所有可用工具名称
            available_tools = [t.name for t in self.tools]
            logger.warning(f"[ReactAgent] 未找到工具: {tool}，可用工具: {available_tools}")
            
            # 尝试模糊匹配
            if available_tools:
                matches = difflib.get_close_matches(tool, available_tools, n=1, cutoff=0.5)
                if matches:
                    suggested_tool = matches[0]
                    logger.info(f"[ReactAgent] 发现相似工具: {suggested_tool}，尝试使用")
                    tool_func = next((t for t in self.tools if t.name == suggested_tool), None)
            
            if not tool_func:
                return f"未找到工具: '{tool}'。可用工具列表: {', '.join(available_tools)}"
        
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