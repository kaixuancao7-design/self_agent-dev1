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
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=self.tools,
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
                for chunk in self.agent.stream({"messages": messages}, stream_mode="values", context=context):
                    yielded_any = True
                    yield chunk
                return
            except Exception as exc:
                logger.error(
                    f"[ReactAgent] 模型流式调用失败，第 {attempt} 次重试，错误：{exc}"
                )
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
        iteration = 0

        while True:
            iteration += 1
            self.session.log("observe", {"iteration": iteration, "messages": messages, "context": context})
            self.session.log("think", {"iteration": iteration, "note": "开始模型执行"})

            last_output = ""
            last_yielded = ""
            tool_calls = []

            for chunk in self._stream_with_retries(messages, context):
                if not isinstance(chunk, dict):
                    continue
                messages_in_chunk = chunk.get("messages")
                if not messages_in_chunk:
                    continue

                latest_message = messages_in_chunk[-1]
                if latest_message.content is not None:
                    content = latest_message.content.strip()
                    if content != last_yielded:
                        if content.startswith(last_yielded):
                            delta = content[len(last_yielded) :]
                        else:
                            delta = content
                        last_yielded = content
                        self.session.log("stream_chunk", {"iteration": iteration, "chunk": delta})
                        yield delta
                    last_output = content

                tool_calls = getattr(latest_message, "tool_calls", []) or []

            self.session.log(
                "result",
                {
                    "iteration": iteration,
                    "output": last_output,
                    "tool_calls": [tc.get("name") for tc in tool_calls],
                },
            )

            messages.append({"role": "assistant", "content": last_output})

            if tool_calls and iteration < max_iterations:
                self.session.log(
                    "reflect",
                    {
                        "iteration": iteration,
                        "action": "continue",
                        "reason": "tool_calls_detected",
                        "tools": [tc.get("name") for tc in tool_calls],
                    },
                )
                continue

            stop_reason = "no_tool_calls" if not tool_calls else "max_iterations_reached"
            self.session.log(
                "reflect",
                {
                    "iteration": iteration,
                    "action": "stop",
                    "reason": stop_reason,
                    "tool_calls": [tc.get("name") for tc in tool_calls],
                },
            )
            break

    def stream_to_text(self, query: str, max_iterations: int = 3) -> str:
        return "".join(self.execute_stream(query, max_iterations=max_iterations))



if __name__ == "__main__":
    react_agent = ReactAgent()
    for chunk in react_agent.execute_stream("给我生成我的使用报告"):
        print(chunk,end="",flush=True)