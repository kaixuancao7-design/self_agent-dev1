"""
日志保存相关函数
"""

import logging
from datetime import datetime
from utils.path_tool import get_abs_path
import os

LOG_ROOT = get_abs_path("logs")

os.makedirs(LOG_ROOT, exist_ok=True)


DEFAULT_LOG_FORMAT = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
DEFAULT_LOG_FILE = os.path.join(LOG_ROOT, f"agent_log_{datetime.now().strftime('%Y%m%d_%H')}.log")


def get_logger(
        name: str = "agent_logger", 
        log_file: str = DEFAULT_LOG_FILE, 
        log_format: logging.Formatter = DEFAULT_LOG_FORMAT,
        file_level: int = logging.DEBUG,
        console_level: int = logging.INFO
        ) -> logging.Logger:
    """
    获取一个配置好的日志记录器。
    :param name: 日志记录器的名称
    :param log_file: 日志文件的路径，默认为logs目录下以当前时间命名的文件
    :param log_format: 日志格式化器，默认为DEFAULT_LOG_FORMAT
    :return: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    #避免重复添加处理器
    if logger.handlers:
        return logger

    # 创建文件处理器
    if not os.path.exists(log_file):
        open(log_file, 'a').close()  # 创建空文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(log_format)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_format)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



logger = get_logger()