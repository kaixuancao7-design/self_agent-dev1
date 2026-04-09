from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langchain.agents.middleware import AgentState, before_agent, after_agent, before_model, after_model, wrap_model_call, wrap_tool_call



@tool(description="查询天气")
def query_weather():
    return f"南京的天气晴朗，温度25度。"

@tool(description="获取股价，传入股票名称，返回字符串信息")
def query_stock(name:str) -> str:
    return f"{name}股票价格稳定，最新价格为100元。"

@tool(description="获取股票描述，传入股票名称，返回字符串信息")
def query_news(name:str) -> str:
    return f"{name}公司发布了新的产品，市场反应积极。"

@before_agent
def log_before_agent(state:AgentState , runtime:Runtime)-> None:
    print(f"[before_agent] 当前状态：{len(state['messages'])}条消息")

@after_agent
def log_after_agent(state:AgentState , runtime:Runtime)-> None:
    print(f"[after_agent] 当前状态：{len(state['messages'])}条消息")

@before_model
def log_before_model(state:AgentState , runtime:Runtime)-> None:
    print(f"[before_model] 当前状态：{len(state['messages'])}条消息")
@after_model
def log_after_model(state:AgentState , runtime:Runtime)-> None:
    print(f"[after_model] 当前状态：{len(state['messages'])}条消息")


@wrap_model_call
def model_call_hook(request, handle):
    print(f"[model_call_hook] 模型请求：{request}")
    result = handle(request)
    print(f"[model_call_hook] 模型响应：{result}")
    return result

@wrap_tool_call
def tool_call_hook(request,handle):
    print(f"[tool_call_hook] 工具请求：{request.tool_call['name']}，参数：{request.tool_call['args']}")
    result = handle(request)
    print(f"[tool_call_hook] 工具结果：{result}")
    return result

agent = create_agent(
    model=ChatTongyi(model ="qwen3-max"),
    tools=[query_weather, query_stock, query_news],
    system_prompt="你是一个股票分析师，可以提供股票相关的信息，记住告知我思考过程，让我知道你调用的那个工具。",
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, tool_call_hook]

)
for chunk in agent.stream(
    {
        "messages": [
            {"role": "user","content": "传智教育股价是多少，请介绍一下"},

        ]
    },
    stream_mode="values",
):
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"模型回复：{type(latest_message).__name__}, {latest_message.content}")

    try:
        if latest_message.tool_calls:
            print(f"工具调用：{[tc['name'] for tc in latest_message.tool_calls]}")
    except AttributeError as e:
        pass