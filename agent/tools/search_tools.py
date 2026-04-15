"""
网络搜索工具
包含网络搜索相关功能
"""
from langchain_core.tools import tool
from utils.logger_handler import logger
import requests
import json


class WebSearchService:
    """
    网络搜索服务
    提供网络搜索能力
    """
    
    def __init__(self):
        self.search_url = "https://www.googleapis.com/customsearch/v1"
        self.api_key = None  # 需要在配置中设置
        self.cx = None  # 自定义搜索引擎ID
    
    def search(self, query: str, max_results: int = 5) -> list:
        """
        执行网络搜索
        
        :param query: 搜索查询词
        :param max_results: 最大返回结果数
        :return: 搜索结果列表
        """
        results = []
        
        # 如果没有配置API key，使用模拟搜索
        if not self.api_key or not self.cx:
            logger.info(f"[WebSearch] 使用模拟搜索: {query}")
            return self._mock_search(query, max_results)
        
        try:
            params = {
                "q": query,
                "key": self.api_key,
                "cx": self.cx,
                "num": max_results
            }
            response = requests.get(self.search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": item.get("displayLink", "")
                    })
            
            logger.info(f"[WebSearch] 搜索完成，获取到 {len(results)} 条结果")
            return results
        except Exception as e:
            logger.error(f"[WebSearch] 搜索失败: {e}", exc_info=True)
            return self._mock_search(query, max_results)
    
    def _mock_search(self, query: str, max_results: int) -> list:
        """
        模拟搜索结果（用于测试）
        
        :param query: 搜索查询词
        :param max_results: 最大返回结果数
        :return: 模拟搜索结果列表
        """
        mock_results = [
            {
                "title": f"关于{query}的最新资讯",
                "link": "https://example.com/news",
                "snippet": f"这是关于{query}的最新新闻报道，包含最新动态和分析。",
                "source": "example.com"
            },
            {
                "title": f"{query} - 维基百科",
                "link": "https://example.com/wiki",
                "snippet": f"{query}是一个重要的概念，在多个领域有广泛应用。",
                "source": "wikipedia.org"
            },
            {
                "title": f"{query}行业分析报告",
                "link": "https://example.com/report",
                "snippet": f"最新的{query}行业分析报告，涵盖市场趋势和预测。",
                "source": "research.com"
            },
            {
                "title": f"{query}技术博客",
                "link": "https://example.com/blog",
                "snippet": f"深入探讨{query}的技术细节和最佳实践。",
                "source": "techblog.com"
            },
            {
                "title": f"{query}相关视频",
                "link": "https://example.com/video",
                "snippet": f"观看{query}的视频教程和演示。",
                "source": "youtube.com"
            }
        ]
        
        return mock_results[:max_results]


# 全局实例
web_search_service = WebSearchService()


@tool(description="进行网络搜索，获取最新的信息和新闻")
def web_search(query: str, max_results: int = 5) -> str:
    """
    进行网络搜索
    
    :param query: 搜索查询词
    :param max_results: 最大返回结果数（默认5条）
    :return: JSON格式的搜索结果列表
    """
    logger.info(f"[web_search] 执行搜索: {query}")
    try:
        results = web_search_service.search(query, max_results)
        result_str = json.dumps(results, ensure_ascii=False, indent=2)
        logger.info(f"[web_search] 搜索完成，结果长度: {len(result_str)}")
        return result_str
    except Exception as e:
        logger.error(f"[web_search] 搜索失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool(description="获取指定URL的网页内容")
def fetch_webpage(url: str) -> str:
    """
    获取指定URL的网页内容
    
    :param url: 网页URL
    :return: 网页内容摘要
    """
    logger.info(f"[fetch_webpage] 获取网页内容: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # 提取文本内容
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 获取标题
        title = soup.title.string if soup.title else "无标题"
        
        # 获取正文内容
        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text() for p in paragraphs[:5]])
        
        # 如果内容太长，截断
        if len(content) > 500:
            content = content[:500] + "..."
        
        result = {
            "title": title,
            "url": url,
            "content": content
        }
        
        logger.info(f"[fetch_webpage] 获取成功，内容长度: {len(content)}")
        return json.dumps(result, ensure_ascii=False)
    except ImportError:
        logger.warning("[fetch_webpage] BeautifulSoup未安装，返回原始HTML")
        return json.dumps({
            "url": url,
            "content": "需要安装BeautifulSoup来解析网页内容"
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[fetch_webpage] 获取失败: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)
