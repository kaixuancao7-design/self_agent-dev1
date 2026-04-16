"""
Firecrawl Skill - 网页内容抓取
基于 Firecrawl MCP 提供网页内容抓取和处理能力
"""
import json
import os
from datetime import datetime
from langchain_core.tools import tool
from utils.logger_handler import logger

# 尝试导入 Firecrawl SDK
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    logger.warning("[Firecrawl] firecrawl-py 未安装，请运行: pip install firecrawl-py")


class FirecrawlClient:
    """Firecrawl 客户端"""
    
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化 Firecrawl 客户端"""
        if not FIRECRAWL_AVAILABLE:
            return
        
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if api_key:
            try:
                self.client = FirecrawlApp(api_key=api_key)
                logger.info("[Firecrawl] Firecrawl 客户端初始化成功")
            except Exception as e:
                logger.error(f"[Firecrawl] Firecrawl 客户端初始化失败: {e}")
        else:
            logger.warning("[Firecrawl] 未设置 FIRECRAWL_API_KEY 环境变量")


# 创建客户端实例
firecrawl_client = FirecrawlClient()


@tool(description="Firecrawl网页抓取：抓取指定URL的完整内容，支持JavaScript渲染")
def firecrawl_scrape(url: str, output_format: str = "markdown") -> str:
    """
    Firecrawl 网页抓取
    
    :param url: 网页地址
    :param output_format: 输出格式 (markdown/json/html)
    :return: JSON格式的抓取结果
    """
    logger.info(f"[firecrawl_scrape] 抓取网页: {url}, 输出格式: {output_format}")
    
    result = {
        "success": False,
        "url": url,
        "format": output_format,
        "content": "",
        "title": "",
        "generated_at": datetime.now().isoformat()
    }
    
    if not firecrawl_client.client:
        result["error"] = "Firecrawl 客户端未初始化，请设置 FIRECRAWL_API_KEY"
        return json.dumps(result, ensure_ascii=False)
    
    try:
        # 调用 Firecrawl API
        scrape_result = firecrawl_client.client.scrape_url(url, params={"formats": output_format})
        
        if scrape_result and "content" in scrape_result:
            result["success"] = True
            result["content"] = scrape_result.get("content", "")
            result["title"] = scrape_result.get("metadata", {}).get("title", "")
            logger.info(f"[firecrawl_scrape] 抓取成功")
        else:
            result["error"] = "抓取失败，未返回内容"
            
    except Exception as e:
        logger.error(f"[firecrawl_scrape] 抓取失败: {e}")
        result["error"] = str(e)
    
    return json.dumps(result, ensure_ascii=False)


@tool(description="Firecrawl网站爬取：爬取整个网站的多个页面")
def firecrawl_crawl(url: str, limit: int = 10) -> str:
    """
    Firecrawl 网站爬取
    
    :param url: 网站根URL
    :param limit: 爬取页面数量限制
    :return: JSON格式的爬取结果
    """
    logger.info(f"[firecrawl_crawl] 爬取网站: {url}, 限制: {limit}")
    
    result = {
        "success": False,
        "url": url,
        "pages_crawled": 0,
        "results": [],
        "generated_at": datetime.now().isoformat()
    }
    
    if not firecrawl_client.client:
        result["error"] = "Firecrawl 客户端未初始化，请设置 FIRECRAWL_API_KEY"
        return json.dumps(result, ensure_ascii=False)
    
    try:
        # 调用 Firecrawl API
        crawl_result = firecrawl_client.client.crawl_url(url, params={"maxPages": limit})
        
        if crawl_result and "data" in crawl_result:
            result["success"] = True
            result["pages_crawled"] = len(crawl_result["data"])
            result["results"] = crawl_result["data"]
            logger.info(f"[firecrawl_crawl] 爬取成功，共 {result['pages_crawled']} 页")
        else:
            result["error"] = "爬取失败，未返回内容"
            
    except Exception as e:
        logger.error(f"[firecrawl_crawl] 爬取失败: {e}")
        result["error"] = str(e)
    
    return json.dumps(result, ensure_ascii=False)


@tool(description="Firecrawl搜索：使用Firecrawl进行网页搜索")
def firecrawl_search(query: str, limit: int = 5) -> str:
    """
    Firecrawl 网页搜索
    
    :param query: 搜索关键词
    :param limit: 返回结果数量
    :return: JSON格式的搜索结果
    """
    logger.info(f"[firecrawl_search] 搜索: {query}, 限制: {limit}")
    
    result = {
        "success": False,
        "query": query,
        "results": [],
        "generated_at": datetime.now().isoformat()
    }
    
    if not firecrawl_client.client:
        result["error"] = "Firecrawl 客户端未初始化，请设置 FIRECRAWL_API_KEY"
        return json.dumps(result, ensure_ascii=False)
    
    try:
        # 调用 Firecrawl API
        search_result = firecrawl_client.client.search(query, params={"limit": limit})
        
        if search_result and "results" in search_result:
            result["success"] = True
            result["results"] = search_result["results"]
            logger.info(f"[firecrawl_search] 搜索成功，共 {len(result['results'])} 条结果")
        else:
            result["error"] = "搜索失败，未返回内容"
            
    except Exception as e:
        logger.error(f"[firecrawl_search] 搜索失败: {e}")
        result["error"] = str(e)
    
    return json.dumps(result, ensure_ascii=False)


# 模块导出
__all__ = ["firecrawl_scrape", "firecrawl_crawl", "firecrawl_search"]
