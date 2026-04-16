"""
Ollama Provider - 本地Ollama实现

适合完全离线、隐私敏感、无API成本的场景。
"""
import openai
from typing import List, Dict, Any, Optional
from model.base import BaseLLM, BaseEmbedding
from utils.logger_handler import logger


class OllamaLLM(BaseLLM):
    """Ollama API实现 - 通过OpenAI-Compatible模式接入"""
    
    def __init__(self, model: str = "qwen3.5:9b", base_url: str = "http://localhost:11434"):
        self._model_name = model
        self._base_url = base_url
        self._client = openai.OpenAI(
            api_key="ollama",
            base_url=f"{base_url}/v1"
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
            logger.error(f"[OllamaLLM] 聊天请求失败: {e}")
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
            logger.error(f"[OllamaLLM] 流式请求失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取token数量"""
        # Ollama使用的模型通常有不同的tokenizer
        # 这里使用粗略估算
        return len(text) // 4
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding实现"""
    
    def __init__(self, model: str = "all-minilm", base_url: str = "http://localhost:11434"):
        self._model_name = model
        self._base_url = base_url
        self._dimension = 384  # all-minilm默认维度
        self._client = openai.OpenAI(
            api_key="ollama",
            base_url=f"{base_url}/v1"
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
            logger.error(f"[OllamaEmbedding] 向量化失败: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """单文本向量化"""
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._dimension
