"""
Get Weather Skill - 获取指定城市的天气信息
"""
from langchain_core.tools import tool
from utils.logger_handler import logger
from utils.config_handler import agent_cfg
import random


@tool(description="获取指定城市的天气信息")
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    :param city: 城市名称
    :return: 天气信息字符串
    """
    logger.info(f"[get_weather] 获取城市天气: {city}")
    
    # 后续可接入真实天气API
    # 模拟天气数据
    weathers = ["晴天", "多云", "阴天", "小雨", "中雨"]
    weather = random.choice(weathers)
    temp = random.randint(15, 35)
    humidity = random.randint(40, 80)
    wind = random.randint(5, 20)
    
    return f"{city}的天气是{weather}，温度{temp}度，湿度{humidity}%，风速{wind}公里每小时。"


if __name__ == "__main__":
    # 测试
    result = get_weather.invoke({"city": "北京"})
    print(result)
