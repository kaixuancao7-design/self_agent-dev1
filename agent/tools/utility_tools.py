"""
通用工具
包含天气、定位、用户ID、时间等通用功能
"""
from langchain_core.tools import tool
from utils.config_handler import agent_cfg
from utils.logger_handler import logger
import random
from datetime import datetime


@tool(description="获取指定城市的天气信息")
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    :param city: 城市名称
    :return: 天气信息字符串
    """
    logger.info(f"[get_weather] 获取城市天气: {city}")
    # 后续可接入真实天气API
    return f"{city}的天气是晴天，温度25度，湿度60%，风速10公里每小时。"


@tool(description="获取用户定位信息")
def get_user_location() -> str:
    """
    获取用户定位信息
    
    :return: 用户位置字符串
    """
    locations = agent_cfg.get("user_locations", [
        "北京市朝阳区", 
        "上海市浦东新区", 
        "广州市天河区", 
        "深圳市南山区"
    ])
    location = random.choice(locations)
    logger.info(f"[get_user_location] 获取用户位置: {location}")
    return f"用户位于{location}。"


@tool(description="获取用户ID")
def get_user_id() -> str:
    """
    获取用户ID
    
    :return: 用户ID字符串
    """
    user_ids = agent_cfg.get("user_ids", [
        "USER_001", 
        "USER_002", 
        "USER_003", 
        "USER_004", 
        "USER_005"
    ])
    user_id = random.choice(user_ids)
    logger.info(f"[get_user_id] 获取用户ID: {user_id}")
    return user_id


@tool(description="获取当前月份")
def get_current_month() -> str:
    """
    获取当前月份（返回真实时间）
    
    :return: 当前月份字符串
    """
    now = datetime.now()
    month_str = f"{now.year}年{now.month}月"
    logger.info(f"[get_current_month] 获取当前月份: {month_str}")
    return f"当前月份是{month_str}。"
