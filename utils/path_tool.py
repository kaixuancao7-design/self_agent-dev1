"""
为整个项目提供路径相关的工具函数，例如获取项目根目录、构建文件路径等。
"""

import os

def get_project_root() -> str:
    """
    获取项目根目录的绝对路径。
    """
    #当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    #往上跳两级，就是项目根目录的绝对路径
    project_root_path = os.path.dirname(os.path.dirname(current_file_path))
    return project_root_path

def get_abs_path(relative_path: str) -> str:
    """
    将相对路径转换为绝对路径，基于项目根目录。
    """
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path