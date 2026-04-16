"""
DeepSeek Provider - DeepSeek API实现

DeepSeek提供高性能的中英文模型，适合成本优化和特定语言优化场景。
"""
import openai
from typing import List, Dict, Any, Optional
from model.base import BaseLLM, BaseEmbedding
from utils.logger_handler import logger


class DeepSeekLLM(BaseLLM):
    """DeepSeek API实现 - 通过OpenAI-Compatible模式接入"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self._model_name = model
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """核心聊天接口"""
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[DeepSeekLLM] 聊天请求失败: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """简单文本生成接口"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)
    
    def chat_stream(self, messages: List[Dict[str, str]]) -> str:
        """流式聊天接口"""
        try:
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                stream=True
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            return full_response
        except Exception as e:
            logger.error(f"[DeepSeekLLM] 流式请求失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取token数量"""
        # 使用tiktoken估算
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name


class DeepSeekEmbedding(BaseEmbedding):
    """DeepSeek Embedding实现"""
    
    def __init__(self, api_key: str, model: str = "deepseek-embed"):
        self._model_name = model
        self._dimension = 1024
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self._model_name
            )
            return [emb.embedding for emb in response.data]
        except Exception as e:
            logger.error(f"[DeepSeekEmbedding] 向量化失败: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """单文本向量化"""
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._dimension
