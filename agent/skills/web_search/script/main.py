"""
Web Search Skill - 进行网络搜索获取最新信息
"""
import json
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="进行网络搜索，获取最新的信息和新闻")
def web_search(query: str, max_results: int = 5) -> str:
    """
    进行网络搜索，获取最新的信息和新闻
    
    :param query: 搜索关键词
    :param max_results: 返回结果数量，默认5条
    :return: JSON格式的搜索结果列表
    """
    logger.info(f"[web_search] 执行搜索: {query}")
    
    try:
        from agent.tools.search_tools import web_search_service
        
        results = web_search_service.search(query, max_results)
        result_str = json.dumps(results, ensure_ascii=False, indent=2)
        logger.info(f"[web_search] 搜索完成，结果长度: {len(result_str)}")
        return result_str
    except Exception as e:
        logger.error(f"[web_search] 搜索失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 测试
    result = web_search.invoke({"query": "AI Agent 2026 趋势", "max_results": 3})
    print(result)
