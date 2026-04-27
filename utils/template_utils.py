#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模板工具函数 - 提供通用的模板变量替换功能
"""

def replace_template_variables(content: str, context: dict) -> str:
    """
    替换文本中的模板变量
    
    :param content: 包含模板变量的文本内容
    :param context: 上下文变量字典，key为变量名，value为变量值
    :return: 替换后的文本内容
    """
    if not context:
        return content
    
    for key, value in context.items():
        # 支持两种格式：{{{key}}} 和 {{key}}
        content = content.replace(f"{{{{{key}}}}}", str(value))
        content = content.replace(f"{{{key}}}", str(value))
    
    return content