"""
Agent 工具模块

包含：
- utility_tools: 通用工具（天气、定位、用户ID、时间、知识库管理）
- report_tools: 报告相关工具
- advanced_tools: 高级能力工具（评估）
- search_tools: 网络搜索工具

注意：核心工具（rag_summarize、task_decompose、fact_check、web_search）已迁移到 Skills 框架
"""

from .utility_tools import get_user_location, get_user_id, get_current_month, \
    get_knowledge_base_stats, list_databases, create_database, delete_database, \
    switch_database, list_uploaded_files, remove_file_from_knowledge_base, reparse_file_in_knowledge_base
from .report_tools import get_user_history, fill_context_report
from .advanced_tools import evaluate_result
from .search_tools import fetch_webpage

__all__ = [
    "get_user_location",
    "get_user_id",
    "get_current_month",
    "get_knowledge_base_stats",
    "list_databases",
    "create_database",
    "delete_database",
    "switch_database",
    "list_uploaded_files",
    "remove_file_from_knowledge_base",
    "reparse_file_in_knowledge_base",
    "get_user_history",
    "fill_context_report",
    "evaluate_result",
    "fetch_webpage"
]
