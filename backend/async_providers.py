"""
异步模型调用模块

提供异步的 LLM 调用、检索服务、文件解析等功能
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, Generator
from utils.logger_handler import logger


class AsyncOllamaProvider:
    """异步 Ollama 模型调用"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> str:
        """异步生成文本"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if stream:
                    result = ""
                    async for chunk in response.content.iter_chunks():
                        data = chunk[0].decode('utf-8')
                        try:
                            json_data = json.loads(data)
                            if "response" in json_data:
                                result += json_data["response"]
                            if json_data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                    return result
                else:
                    data = await response.json()
                    return data.get("response", "")
    
    async def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """异步聊天"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if stream:
                    result = ""
                    async for chunk in response.content.iter_chunks():
                        data = chunk[0].decode('utf-8')
                        try:
                            json_data = json.loads(data)
                            if "message" in json_data and "content" in json_data["message"]:
                                result += json_data["message"]["content"]
                            if json_data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                    return result
                else:
                    data = await response.json()
                    return data.get("message", {}).get("content", "")
    
    async def embed(self, model: str, prompt: str) -> List[float]:
        """异步生成嵌入向量"""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                return data.get("embedding", [])


class AsyncRAGService:
    """异步 RAG 服务"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索相关文档"""
        # 包装同步调用为异步
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._sync_retrieve, query, top_k)
        return results
    
    def _sync_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """同步检索（内部使用）"""
        try:
            return self.vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.error(f"[AsyncRAG] 检索失败: {e}")
            return []
    
    async def summarize(self, query: str) -> str:
        """异步总结检索结果"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._sync_summarize, query)
        return result
    
    def _sync_summarize(self, query: str) -> str:
        """同步总结（内部使用）"""
        from rag.rag_service import RagSummerizeService
        try:
            rag_service = RagSummerizeService()
            return rag_service.rag_summarize(query)
        except Exception as e:
            logger.error(f"[AsyncRAG] 总结失败: {e}")
            return f"检索失败: {str(e)}"


class AsyncFileProcessor:
    """异步文件处理器"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    async def process_file(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """异步处理文件"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._sync_process_file, file_path, file_name)
        return result
    
    def _sync_process_file(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """同步处理文件（内部使用）"""
        try:
            return self.vector_store.upload_file(file_path, file_name)
        except Exception as e:
            logger.error(f"[AsyncFileProcessor] 文件处理失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def batch_process_files(self, files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """异步批量处理文件"""
        tasks = [
            self.process_file(file["path"], file["name"])
            for file in files
        ]
        results = await asyncio.gather(*tasks)
        return results


class AsyncRagasEvaluator:
    """异步 RAGAS 评估器"""
    
    async def evaluate(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """异步评估 RAG 结果"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._sync_evaluate, query, answer, contexts)
        return result
    
    def _sync_evaluate(self, query: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """同步评估（内部使用）"""
        from rag.ragas_evaluator import evaluate_rag_pipeline
        try:
            return evaluate_rag_pipeline(query, answer, contexts)
        except Exception as e:
            logger.error(f"[AsyncRagas] 评估失败: {e}")
            return {"success": False, "message": str(e)}


# 全局实例
async_ollama = AsyncOllamaProvider()


async def async_chat(model: str, messages: List[Dict[str, str]], stream: bool = False) -> str:
    """便捷函数：异步聊天"""
    return await async_ollama.chat(model, messages, stream)


async def async_embed(model: str, prompt: str) -> List[float]:
    """便捷函数：异步生成嵌入向量"""
    return await async_ollama.embed(model, prompt)