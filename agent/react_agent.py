import time
from typing import Any

from langchain.agents import create_agent
from agent.session import AgentSession
from agent.tools.agent_tools import rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, generate_exaction_data, fill_context_report, get_user_history
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from model.factory import chat_model
from utils.config_handler import agent_cfg
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt


class ReactAgent:
    def __init__(self):
        self.tools = [
            rag_summarize,
            get_weather,
            get_user_location,
            get_user_id,
            get_current_month,
            generate_exaction_data,
            fill_context_report,
            get_user_history,
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



if __name__ == "__main__":
    react_agent = ReactAgent()
    for chunk in react_agent.execute_stream("给我生成我的使用报告"):
        print(chunk,end="",flush=True)