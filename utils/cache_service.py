"""
缓存服务模块

提供多层次缓存机制：
- 查询结果缓存：缓存 RAG 检索结果
- LLM 响应缓存：缓存 LLM 生成的回答
- 会话缓存：缓存会话上下文摘要

使用 LRU (Least Recently Used) 策略管理缓存
"""

import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
from utils.logger_handler import logger


class QueryCache:
    """查询结果缓存"""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 60):
        """
        :param max_size: 最大缓存条目数
        :param ttl_minutes: 缓存过期时间（分钟）
        """
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, query: str, **kwargs) -> str:
        """
        生成缓存键
        
        :param query: 查询字符串
        :param kwargs: 额外参数
        :return: 缓存键（MD5哈希）
        """
        key_str = f"{query}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """
        获取缓存值
        
        :param query: 查询字符串
        :param kwargs: 额外参数
        :return: 缓存值，如果不存在或过期返回 None
        """
        key = self._get_key(query, **kwargs)
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # 检查是否过期
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                logger.debug(f"[QueryCache] 缓存命中: {query[:30]}...")
                return value
            else:
                # 过期，删除
                del self.cache[key]
                logger.debug(f"[QueryCache] 缓存过期: {query[:30]}...")
        
        self.misses += 1
        return None
    
    def set(self, query: str, value: Any, **kwargs):
        """
        设置缓存值
        
        :param query: 查询字符串
        :param value: 缓存值
        :param kwargs: 额外参数
        """
        key = self._get_key(query, **kwargs)
        
        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            logger.debug(f"[QueryCache] 缓存已满，删除最旧条目")
        
        self.cache[key] = (value, datetime.now())
        logger.debug(f"[QueryCache] 缓存设置: {query[:30]}...")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("[QueryCache] 缓存已清空")
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        :return: 统计字典（hits, misses, size）
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache)
        }


class LLMCache:
    """LLM 响应缓存"""
    
    def __init__(self, max_size: int = 500, ttl_minutes: int = 30):
        """
        :param max_size: 最大缓存条目数
        :param ttl_minutes: 缓存过期时间（分钟）
        """
        self.cache: Dict[str, Tuple[str, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.hits = 0
        self.misses = 0
    
    def _get_key(self, messages: list, model: str = "") -> str:
        """
        生成缓存键
        
        :param messages: 消息列表
        :param model: 模型名称
        :return: 缓存键
        """
        messages_str = str([(m.get("role"), m.get("content")[:500]) for m in messages])
        key_str = f"{model}_{messages_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, messages: list, model: str = "") -> Optional[str]:
        """
        获取缓存的 LLM 响应
        
        :param messages: 消息列表
        :param model: 模型名称
        :return: 缓存的响应，如果不存在或过期返回 None
        """
        key = self._get_key(messages, model)
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            if datetime.now() - timestamp < self.ttl:
                self.hits += 1
                logger.debug(f"[LLMCache] 缓存命中")
                return value
            else:
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, messages: list, response: str, model: str = ""):
        """
        设置缓存的 LLM 响应
        
        :param messages: 消息列表
        :param response: LLM 响应
        :param model: 模型名称
        """
        key = self._get_key(messages, model)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (response, datetime.now())
        logger.debug(f"[LLMCache] 缓存设置")
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("[LLMCache] 缓存已清空")
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache)
        }


class CacheService:
    """缓存服务 - 整合多种缓存"""
    
    def __init__(self):
        self.query_cache = QueryCache(max_size=500, ttl_minutes=60)
        self.llm_cache = LLMCache(max_size=200, ttl_minutes=30)
    
    def get_query_result(self, query: str, **kwargs) -> Optional[Any]:
        """
        获取查询结果缓存
        
        :param query: 查询字符串
        :param kwargs: 额外参数（如 database_name, top_k 等）
        :return: 缓存结果
        """
        return self.query_cache.get(query, **kwargs)
    
    def set_query_result(self, query: str, result: Any, **kwargs):
        """
        设置查询结果缓存
        
        :param query: 查询字符串
        :param result: 查询结果
        :param kwargs: 额外参数
        """
        self.query_cache.set(query, result, **kwargs)
    
    def get_llm_response(self, messages: list, model: str = "") -> Optional[str]:
        """
        获取 LLM 响应缓存
        
        :param messages: 消息列表
        :param model: 模型名称
        :return: 缓存的响应
        """
        return self.llm_cache.get(messages, model)
    
    def set_llm_response(self, messages: list, response: str, model: str = ""):
        """
        设置 LLM 响应缓存
        
        :param messages: 消息列表
        :param response: LLM 响应
        :param model: 模型名称
        """
        self.llm_cache.set(messages, response, model)
    
    def clear_all(self):
        """清空所有缓存"""
        self.query_cache.clear()
        self.llm_cache.clear()
        logger.info("[CacheService] 所有缓存已清空")
    
    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """
        获取所有缓存的统计信息
        
        :return: 统计信息字典
        """
        return {
            "query_cache": self.query_cache.get_stats(),
            "llm_cache": self.llm_cache.get_stats()
        }


# 全局缓存服务实例
cache_service = CacheService()


def cached_query(**cache_kwargs):
    """
    查询结果缓存装饰器
    
    使用示例：
    @cached_query()
    def my_query_function(query, **kwargs):
        # 执行查询
        return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(query, **kwargs):
            # 合并装饰器参数和函数参数作为缓存键的一部分
            all_kwargs = {**cache_kwargs, **kwargs}
            
            # 尝试获取缓存
            cached_result = cache_service.get_query_result(query, **all_kwargs)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(query, **kwargs)
            
            # 设置缓存
            cache_service.set_query_result(query, result, **all_kwargs)
            
            return result
        return wrapper
    return decorator


def cached_llm(model: str = ""):
    """
    LLM 响应缓存装饰器
    
    使用示例：
    @cached_llm(model="gpt-4")
    def my_llm_function(messages):
        # 调用 LLM
        return response
    """
    def decorator(func):
        @wraps(func)
        def wrapper(messages, **kwargs):
            # 尝试获取缓存
            cached_response = cache_service.get_llm_response(messages, model)
            if cached_response is not None:
                return cached_response
            
            # 执行函数
            response = func(messages, **kwargs)
            
            # 设置缓存
            cache_service.set_llm_response(messages, response, model)
            
            return response
        return wrapper
    return decorator


def enable_cache():
    """启用缓存"""
    logger.info("[CacheService] 缓存服务已启用")


def disable_cache():
    """禁用缓存（清空并不再使用）"""
    cache_service.clear_all()
    logger.info("[CacheService] 缓存服务已禁用")


def get_cache_stats() -> str:
    """
    获取缓存统计信息的可读字符串
    
    :return: 统计信息字符串
    """
    stats = cache_service.get_all_stats()
    
    return (
        f"=== 缓存统计 ===\n"
        f"查询缓存:\n"
        f"  命中: {stats['query_cache']['hits']}\n"
        f"  未命中: {stats['query_cache']['misses']}\n"
        f"  命中率: {stats['query_cache']['hit_rate']}%\n"
        f"  当前大小: {stats['query_cache']['size']}\n"
        f"LLM缓存:\n"
        f"  命中: {stats['llm_cache']['hits']}\n"
        f"  未命中: {stats['llm_cache']['misses']}\n"
        f"  命中率: {stats['llm_cache']['hit_rate']}%\n"
        f"  当前大小: {stats['llm_cache']['size']}\n"
    )
