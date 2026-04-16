"""
RAG Summarize Skill - 检索知识库并返回格式化内容
"""
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="检索知识库中的相关文档并返回格式化内容")
def rag_summarize(query: str) -> str:
    """
    检索知识库中的相关文档并返回格式化的文档内容
    
    :param query: 用户的查询问题
    :return: 格式化的参考文档内容
    """
    logger.info(f"[rag_summarize] 执行检索: {query}")
    
    # 延迟导入避免循环依赖
    from rag.rag_service import RagSummerizeService
    
    try:
        rag_service = RagSummerizeService()
        result = rag_service.rag_summarize(query)
        logger.info(f"[rag_summarize] 检索完成，结果长度: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"[rag_summarize] 检索失败: {e}", exc_info=True)
        return f"检索失败：{str(e)}"


if __name__ == "__main__":
    # 测试
    result = rag_summarize.invoke({"query": "AI Agent 最近的热点趋势"})
    print(result)
