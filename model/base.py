"""
统一接口层 - 抽象基类定义

提供统一的LLM和Embedding调用接口，屏蔽不同Provider的实现差异。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class BaseLLM(ABC):
    """
    LLM抽象基类 - 统一不同Provider的调用接口
    
    无论底层使用 Azure OpenAI、OpenAI、DeepSeek 还是 Ollama，
    上层调用代码保持一致。
    """
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        核心聊天接口
        
        :param messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
        :return: 模型响应文本
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        简单文本生成接口
        
        :param prompt: 输入提示词
        :return: 生成的文本
        """
        pass
    
    @abstractmethod
    def chat_stream(self, messages: List[Dict[str, str]]) -> str:
        """
        流式聊天接口
        
        :param messages: 消息列表
        :return: 流式响应（逐段返回）
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        获取文本token数量
        
        :param text: 输入文本
        :return: token数量
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """获取模型名称"""
        pass


class BaseEmbedding(ABC):
    """
    Embedding抽象基类 - 统一不同Provider的调用接口
    
    统一处理批量请求与维度归一化。
    """
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化
        
        :param texts: 文本列表
        :return: 向量列表，每个向量为float数组
        """
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """
        单文本向量化
        
        :param text: 输入文本
        :return: 向量表示
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """获取向量维度"""
        pass


class BaseVisionLLM(ABC):
    """
    Vision LLM抽象基类 - 支持文本+图片的多模态输入
    
    针对图像描述生成（Image Captioning）需求。
    """
    
    @abstractmethod
    def chat_with_image(self, messages: List[Dict[str, str]], 
                        image_urls: Optional[List[str]] = None) -> str:
        """
        多模态聊天接口
        
        :param messages: 消息列表
        :param image_urls: 图片URL列表
        :return: 模型响应文本
        """
        pass
