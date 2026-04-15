"""
核心业务工具
包含与知识库交互的核心功能
"""
from langchain_core.tools import tool
from rag.rag_service import RagSummerizeService
from utils.logger_handler import logger

# 初始化RAG服务
rag = RagSummerizeService()


@tool(description="从向量库中检索相关文档，并根据检索到的文档和用户问题生成总结报告")
def rag_summarize(query: str) -> str:
    """
    RAG总结服务：用户提问，搜索参考资料，将参考资料和用户问题一起输入模型，生成总结报告
    
    :param query: 用户查询词
    :return: 总结报告内容
    """
    try:
        result = rag.rag_summarize(query)
        logger.info(f"[rag_summarize] 执行成功，结果长度: {len(str(result))}")
        return result
    except Exception as e:
        logger.error(f"[rag_summarize] 执行失败: {e}", exc_info=True)
        raise
