from typing import Any

from langchain.agents import create_agent
from agent.tools.agent_tools import rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, generate_exaction_data, fill_context_report,get_user_history
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt, load_rag_prompt, load_report_prompt
from utils.config_handler import agent_cfg
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
import os



class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompt(),
            tools = [rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, 
                     generate_exaction_data,fill_context_report,get_user_history],
            middleware = [log_before_model ,monitor_tool, report_prompt_switch],
        )


    def execute_stream(self,query: str):
        input_dict: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        for chunk in self.agent.stream(input_dict,stream_mode="values",context = {"report": False}):#在context中添加report标记，初始值为False，表示还没有生成报告
            messages = chunk.get("messages")
            if not messages:
                continue
            latest_message = chunk["messages"][-1]#获取最新的模型回复
            if latest_message.content:
                yield latest_message.content.strip()+'\n'#返回最新回复的内容，前端可以根据这个内容进行增量更新



if __name__ == "__main__":
    react_agent = ReactAgent()
    for chunk in react_agent.execute_stream("给我生成我的使用报告"):
        print(chunk,end="",flush=True)