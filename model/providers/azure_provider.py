"""
Azure OpenAI Provider - Azure OpenAI API实现
"""
import openai
from typing import List, Dict, Any, Optional
from model.base import BaseLLM, BaseEmbedding, BaseVisionLLM
from utils.logger_handler import logger


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI API实现"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        self._model_name = deployment_name
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-15-preview"
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
            logger.error(f"[AzureOpenAILLM] 聊天请求失败: {e}")
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
            logger.error(f"[AzureOpenAILLM] 流式请求失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取token数量"""
        # Azure OpenAI使用tiktoken进行token计数
        import tiktoken
        encoding = tiktoken.encoding_for_model(self._model_name)
        return len(encoding.encode(text))
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name


class AzureOpenAIEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding实现"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        self._model_name = deployment_name
        self._dimension = 1536
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-15-preview"
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
            logger.error(f"[AzureOpenAIEmbedding] 向量化失败: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """单文本向量化"""
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._dimension


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision实现 - 支持多模态输入"""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        self._model_name = deployment_name
        self._client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-15-preview"
        )
    
    def chat_with_image(self, messages: List[Dict[str, str]], 
                        image_urls: Optional[List[str]] = None) -> str:
        """多模态聊天接口"""
        try:
            # 构建包含图片的消息
            processed_messages = []
            for msg in messages:
                if msg["role"] == "user" and image_urls:
                    content = []
                    content.append({"type": "text", "text": msg["content"]})
                    for image_url in image_urls:
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
                    processed_messages.append({"role": "user", "content": content})
                else:
                    processed_messages.append(msg)
            
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=processed_messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[AzureVisionLLM] 多模态请求失败: {e}")
            raise
