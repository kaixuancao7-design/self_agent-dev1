"""
模型模块 - 统一接口层

提供统一的LLM和Embedding调用接口，屏蔽不同Provider的实现差异。

核心组件:
- BaseLLM: LLM抽象基类
- BaseEmbedding: Embedding抽象基类
- BaseVisionLLM: Vision LLM抽象基类
- LLMFactory: LLM工厂类
- EmbeddingFactory: Embedding工厂类
- VisionLLMFactory: Vision LLM工厂类

支持的Provider类型:
- azure: Azure OpenAI
- openai: OpenAI原生
- deepseek: DeepSeek
- ollama: Ollama本地部署
"""
from .base import BaseLLM, BaseEmbedding, BaseVisionLLM
from .factory import LLMFactory, EmbeddingFactory, VisionLLMFactory

__all__ = [
    "BaseLLM",
    "BaseEmbedding",
    "BaseVisionLLM",
    "LLMFactory",
    "EmbeddingFactory",
    "VisionLLMFactory"
]
