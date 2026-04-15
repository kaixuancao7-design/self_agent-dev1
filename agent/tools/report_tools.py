"""
报告相关工具
包含用户历史记录查询和报告生成支持功能
"""
from langchain_core.tools import tool
from utils.config_handler import agent_cfg
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
import os


class ExternalDataLoader:
    """
    外部数据加载器（线程安全）
    负责加载和管理用户历史交互记录
    """
    
    def __init__(self):
        self._data = {}
        self._loaded = False
    
    def load(self):
        """加载外部数据"""
        if self._loaded:
            return
        
        external_data_path = get_abs_path(agent_cfg["external_data_path"])
        
        if not os.path.exists(external_data_path):
            logger.warning(f"[外部数据加载]数据文件不存在：{external_data_path}")
            return
        
        try:
            with open(external_data_path, 'r', encoding='utf-8') as f:
                for line in f.readlines()[1:]:  # 跳过第一行表头
                    arr = line.strip().split(",")
                    user_id = arr[0].replace("''", "")
                    features = arr[1].replace("''", "")
                    efficiency = arr[2].replace("''", "")
                    consumables = arr[3].replace("''", "")
                    comparison = arr[4].replace("''", "")
                    time = arr[5].replace("''", "")
                    
                    if user_id not in self._data:
                        self._data[user_id] = {}
                    self._data[user_id][time] = {
                        "特征": features,
                        "清洁效率": efficiency,
                        "耗材": consumables,
                        "对比": comparison,
                        "时间": time
                    }
            self._loaded = True
            logger.info(f"[外部数据加载]成功加载 {len(self._data)} 个用户的数据")
        except Exception as e:
            logger.error(f"[外部数据加载]加载失败: {e}", exc_info=True)
    
    def get_user_data(self, user_id: str, month: str) -> dict:
        """
        获取用户指定月份的数据
        
        :param user_id: 用户ID
        :param month: 月份（格式：YYYY年MM月）
        :return: 用户数据字典
        """
        self.load()
        return self._data.get(user_id, {}).get(month, {})


# 全局实例
data_loader = ExternalDataLoader()


@tool(description="从外部系统中获取用户的历史交互记录")
def get_user_history(user_id: str, month: str) -> str:
    """
    从外部系统中获取用户的历史交互记录
    
    :param user_id: 用户ID
    :param month: 月份（格式：YYYY年MM月）
    :return: 用户历史记录字符串
    """
    logger.info(f"[get_user_history] 获取用户{user_id}在{month}的历史记录")
    result = data_loader.get_user_data(user_id, month)
    if result:
        return str(result)
    logger.warning(f"[get_user_history] 未找到用户{user_id}在{month}的记录")
    return f"没有找到用户{user_id}在{month}的交互记录"


@tool(description="报告提示词切换事件")
def fill_context_report():
    """
    报告提示词切换事件
    
    用于通知系统切换到报告生成模式，中间件会根据此调用切换提示词
    """
    logger.info(f"[fill_context_report] 报告提示词切换事件被调用")
    return "fill_context_report工具函数被调用，提示词切换事件已报告。"
