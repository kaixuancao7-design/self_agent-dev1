 
from langchain.agents.middleware import AgentState, Runtime, dynamic_prompt,wrap_tool_call,before_model,after_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import BaseMessage,ToolMessage
from langgraph.types import Command
from typing import Callable,Any
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompt,load_report_prompt,load_system_prompt


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handle:Callable[[ToolCallRequest],ToolMessage|Command],
)->ToolMessage|Command:
    """
    监控工具调用的中间件函数，打印工具调用的输入和输出
    """
    logger.info(f"[tool monitor]：执行工具 {request.tool_call['name']}")
    logger.info(f"[tool monitor]：输入参数：{request.tool_call['args']}")
    try:
        response = handle(request)
        logger.info(f"[tool monitor]：工具 {request.tool_call['name']} 执行完成")
        if request.tool_call['name'] == "fill_context_report":
            request.runtime.context["report"] = True#如果调用了生成报告的工具，在runtime的context中添加一个标记，表示已经生成了报告，后续可以在模型调用的中间件函数中根据这个标记进行特殊处理

        return response
    except Exception as e:
        logger.error(f"[tool monitor]：工具 {request.tool_call['name']} 执行出错：{e}")
        raise e
    
    
@before_model
def log_before_model(
    state:AgentState,           #整个agent的状态，包括历史对话、工具调用记录等
    runtime:Runtime,            #记录整个执行过程中的上下文信息，可以在中间件函数中添加自定义的上下文信息
):#在模型调用之前执行的中间件函数，打印模型调用的输入
    logger.info(f"[log before model]：即将调用模型，带有{len(state['messages'])}轮历史对话")
    logger.debug(f"[log before model]：{type(state['messages'][-1]).__name__}|历史对话内容：{state['messages'][-1].content.strip()}")

    return None

@dynamic_prompt#在模型调用的中间件函数中动态修改提示词，根据runtime.context中的标记判断是否需要切换提示词
def report_prompt_switch(request: ToolCallRequest):#动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:#是报告生成的场景，切换到报告提示词
        logger.info(f"[report prompt switch]：检测到生成报告的标记，正在切换提示词")
        return load_report_prompt()
    else:#不是报告生成的场景，使用默认的提示词
        return load_system_prompt()
