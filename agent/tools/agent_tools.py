"""
agent工具函数
"""
from langchain_core.tools import tool
from rag.rag_service import RagSummerizeService
import random
from utils.config_handler import agent_cfg
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
import os


rag = RagSummerizeService()
userids = ["USER_001", "USER_002", "USER_003", "USER_004", "USER_005"]
month = ["2025年1月", "2025年2月", "2025年3月", "2025年4月", "2025年5月", "2025年6月", "2025年7月", "2025年8月", "2025年9月", "2025年10月", "2025年11月", "2025年12月"]

@tool(description="从向量库中检索相关文档，并根据检索到的文档和用户问题生成总结报告")
def rag_summarize(query: str) -> str:
    """
    rag总结服务：用户提问，搜索参考资料，将参考资料和用户问题一起输入模型，生成总结报告
    """
    # 1.检索相关文档
    # 2.将文档内容和用户问题一起输入模型，生成总结报告
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气信息")
def get_weather(city: str) -> str:
    """
    获取天气信息的工具函数，暂时返回固定字符串，后续可以接入天气API获取真实天气信息
    """
    return f"{city}的天气是晴天，温度25度，湿度60%，风速10公里每小时。" 


@tool(description="获取用户定位信息")
def get_user_location() -> str:
    """
    获取用户定位信息的工具函数，暂时返回固定字符串，后续可以接入定位API获取真实位置信息
    """
    location = random.choice(["北京市朝阳区", "上海市浦东新区", "广州市天河区", "深圳市南山区"])
    return f"用户位于{location}。"

@tool(description="获取用户ID")
def get_user_id() -> str:
    """
    获取用户ID的工具函数，暂时返回固定字符串，后续可以接入用户认证系统获取真实用户ID
    """
    useid = random.choice(userids)
    return useid

@tool(description="获取当前月份")
def get_current_month() -> str:
    """
    获取当前月份的工具函数，暂时返回固定字符串，后续可以接入时间API获取真实时间信息
    """
    month_str = random.choice(month)
    return f"当前月份是{month_str}。"

exaction_data ={}
def generate_exaction_data():
    """
    {
        "user_id": {
            "month": {"特征": "",
            "清洁效率": "",
            "耗材": "",
            "对比": "",
            "时间": ""},
            "month": {"特征": "",
            "清洁效率": "",
            "耗材": "",
            "对比": "",
            "时间": ""},
             ...
        }
    }
    """
    if not exaction_data:
        external_data_path = get_abs_path(agent_cfg["external_data_path"])

        if not os.path.exists(external_data_path):
            logger.warning(f"[生成工具函数执行数据]外部数据文件不存在：{external_data_path}，无法生成工具函数执行数据")
            return
        with open(external_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:#跳过第一行表头
                arr:list[str] = line.strip().split(",")
                user_id:str = arr[0].replace("''", "")#去掉引号
                features:str = arr[1].replace("''", "")#去掉引号
                efficiency:str = arr[2].replace("''", "")#去掉引号
                consumables:str = arr[3].replace("''", "")#去掉引号
                comparison:str = arr[4].replace("''", "")#去掉引号
                time:str = arr[5].replace("''", "")#去掉引号
                
                if user_id not in exaction_data:
                    exaction_data[user_id] = {}
                exaction_data[user_id][time] = {
                    "特征": features,
                    "清洁效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                    "时间": time
                }
        


@tool(description="从外部系统中获取用户的历史交互记录")
def get_user_history(user_id: str, month: str) -> str:
    """
    从外部系统中获取用户的历史交互记录
    """
    generate_exaction_data()
    try:
        return exaction_data[user_id][month]
    except KeyError:
        logger.warning(f"[获取用户历史交互记录]没有找到用户{user_id}在{month}的交互记录")
        return f"没有找到用户{user_id}在{month}的交互记录"



@tool(description="报告提示词切换事件,供模型调用，暂时不做任何处理，后续可以在这里添加日志记录、统计分析等功能")
def fill_context_report():
    """
    报告提示词切换事件的工具函数，暂时不做任何处理，后续可以在这里添加日志记录、统计分析等功能
    """
    return "fill_context_report工具函数被调用，提示词切换事件已报告。"