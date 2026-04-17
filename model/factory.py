"""
统一模型工厂 - 基于配置自动选择Provider

支持的Provider类型:
- azure: Azure OpenAI
- openai: OpenAI原生
- deepseek: DeepSeek
- ollama: Ollama本地部署
"""
from typing import Optional
from model.base import BaseLLM, BaseEmbedding, BaseVisionLLM
from utils.logger_handler import logger
from utils.config_handler import agent_cfg, rag_cfg


class LLMFactory:
    """LLM工厂类 - 根据配置创建对应的LLM实例"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def create(provider: str, **kwargs) -> BaseLLM:
        """
        创建LLM实例
        
        :param provider: Provider类型 (azure/openai/deepseek/ollama)
        :param kwargs: 配置参数
        :return: BaseLLM实例
        """
        provider = provider.strip().lower()
        
        if provider == "azure":
            return LLMFactory._create_azure(**kwargs)
        elif provider == "openai":
            return LLMFactory._create_openai(**kwargs)
        elif provider == "deepseek":
            return LLMFactory._create_deepseek(**kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama(**kwargs)
        else:
            raise ValueError(f"不支持的Provider类型: {provider}")
    
    @staticmethod
    def _create_azure(api_key: str, endpoint: str, deployment_name: str, **kwargs) -> BaseLLM:
        """创建Azure OpenAI实例"""
        from model.providers.azure_provider import AzureOpenAILLM
        logger.info(f"[LLMFactory] 创建Azure OpenAI实例: {deployment_name} @ {endpoint}")
        return AzureOpenAILLM(api_key, endpoint, deployment_name)
    
    @staticmethod
    def _create_openai(api_key: str, model: str = "gpt-4o", **kwargs) -> BaseLLM:
        """创建OpenAI实例"""
        from model.providers.openai_provider import OpenAILLM
        logger.info(f"[LLMFactory] 创建OpenAI实例: {model}")
        return OpenAILLM(api_key, model)
    
    @staticmethod
    def _create_deepseek(api_key: str, model: str = "deepseek-chat", **kwargs) -> BaseLLM:
        """创建DeepSeek实例"""
        from model.providers.deepseek_provider import DeepSeekLLM
        logger.info(f"[LLMFactory] 创建DeepSeek实例: {model}")
        return DeepSeekLLM(api_key, model)
    
    @staticmethod
    def _create_ollama(model: str = "qwen3.5:9b", base_url: str = "http://localhost:11434", **kwargs) -> BaseLLM:
        """创建Ollama实例"""
        from model.providers.ollama_provider import OllamaLLM
        logger.info(f"[LLMFactory] 创建Ollama实例: {model} @ {base_url}")
        return OllamaLLM(model, base_url)


class EmbeddingFactory:
    """Embedding工厂类 - 根据配置创建对应的Embedding实例"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingFactory, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def create(provider: str, **kwargs) -> BaseEmbedding:
        """
        创建Embedding实例
        
        :param provider: Provider类型 (azure/openai/deepseek/ollama)
        :param kwargs: 配置参数
        :return: BaseEmbedding实例
        """
        provider = provider.strip().lower()
        
        if provider == "azure":
            return EmbeddingFactory._create_azure(**kwargs)
        elif provider == "openai":
            return EmbeddingFactory._create_openai(**kwargs)
        elif provider == "deepseek":
            return EmbeddingFactory._create_deepseek(**kwargs)
        elif provider == "ollama":
            return EmbeddingFactory._create_ollama(**kwargs)
        else:
            raise ValueError(f"不支持的Provider类型: {provider}")
    
    @staticmethod
    def _create_azure(api_key: str, endpoint: str, deployment_name: str, **kwargs) -> BaseEmbedding:
        """创建Azure OpenAI Embedding实例"""
        from model.providers.azure_provider import AzureOpenAIEmbedding
        logger.info(f"[EmbeddingFactory] 创建Azure Embedding实例: {deployment_name} @ {endpoint}")
        return AzureOpenAIEmbedding(api_key, endpoint, deployment_name)
    
    @staticmethod
    def _create_openai(api_key: str, model: str = "text-embedding-3-small", **kwargs) -> BaseEmbedding:
        """创建OpenAI Embedding实例"""
        from model.providers.openai_provider import OpenAIEmbedding
        logger.info(f"[EmbeddingFactory] 创建OpenAI Embedding实例: {model}")
        return OpenAIEmbedding(api_key, model)
    
    @staticmethod
    def _create_deepseek(api_key: str, model: str = "deepseek-embed", **kwargs) -> BaseEmbedding:
        """创建DeepSeek Embedding实例"""
        from model.providers.deepseek_provider import DeepSeekEmbedding
        logger.info(f"[EmbeddingFactory] 创建DeepSeek Embedding实例: {model}")
        return DeepSeekEmbedding(api_key, model)
    
    @staticmethod
    def _create_ollama(model: str = "all-minilm", base_url: str = "http://localhost:11434", **kwargs) -> BaseEmbedding:
        """创建Ollama Embedding实例"""
        from model.providers.ollama_provider import OllamaEmbedding
        logger.info(f"[EmbeddingFactory] 创建Ollama Embedding实例: {model} @ {base_url}")
        return OllamaEmbedding(model, base_url)


class VisionLLMFactory:
    """Vision LLM工厂类 - 创建多模态模型实例"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionLLMFactory, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def create(provider: str, **kwargs) -> BaseVisionLLM:
        """
        创建Vision LLM实例
        
        :param provider: Provider类型 (azure/openai)
        :param kwargs: 配置参数
        :return: BaseVisionLLM实例
        """
        provider = provider.strip().lower()
        
        if provider == "azure":
            return VisionLLMFactory._create_azure(**kwargs)
        elif provider == "openai":
            return VisionLLMFactory._create_openai(**kwargs)
        else:
            raise ValueError(f"不支持的Vision Provider类型: {provider}")
    
    @staticmethod
    def _create_azure(api_key: str, endpoint: str, deployment_name: str, **kwargs) -> BaseVisionLLM:
        """创建Azure Vision LLM实例"""
        from model.providers.azure_provider import AzureVisionLLM
        logger.info(f"[VisionLLMFactory] 创建Azure Vision实例: {deployment_name} @ {endpoint}")
        return AzureVisionLLM(api_key, endpoint, deployment_name)
    
    @staticmethod
    def _create_openai(api_key: str, model: str = "gpt-4o", **kwargs) -> BaseVisionLLM:
        """创建OpenAI Vision LLM实例"""
        from model.providers.openai_provider import OpenAIVisionLLM
        logger.info(f"[VisionLLMFactory] 创建OpenAI Vision实例: {model}")
        return OpenAIVisionLLM(api_key, model)


# 向后兼容的全局变量
class CloudChatModelFactory:
    """云端模型工厂（兼容旧代码）"""
    def generator(self):
        from model.providers.tongyi_provider import TongyiLLM
        model_name = rag_cfg.get("cloud_chat_model_name", "qwen3-max")
        logger.info(f"[ModelFactory] 初始化云端模型：{model_name}")
        return TongyiLLM(model=model_name)


class OllamaChatModelFactory:
    """Ollama模型工厂（兼容旧代码）"""
    def generator(self):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            logger.warning("[ModelFactory] langchain_ollama 包未安装，尝试使用 langchain_community")
            from langchain_community.chat_models.ollama import ChatOllama

        model_name = agent_cfg.get("model", "qwen3.5:9b")
        base_url = agent_cfg.get("base_url", "http://localhost:11434")
        logger.info(f"[ModelFactory] 初始化本地 Ollama 模型：{model_name} @ {base_url}")
        return ChatOllama(model=model_name, base_url=base_url)


class ChatModelFactory:
    """聊天模型工厂（兼容旧代码）"""
    def generator(self):
        source = str(agent_cfg.get("model_source", "cloud")).strip().lower()
        if source in ("ollama", "local"):
            try:
                return OllamaChatModelFactory().generator()
            except Exception:
                if agent_cfg.get("fallback_to_cloud", True):
                    logger.warning(
                        "[ModelFactory] 本地 Ollama 模型加载失败，回退到云端模型。"
                    )
                    return CloudChatModelFactory().generator()
                raise
        return CloudChatModelFactory().generator()


class EmbeddingsFactory:
    """嵌入模型工厂（兼容旧代码）"""
    def generator(self):
        from langchain_community.embeddings import DashScopeEmbeddings
        return DashScopeEmbeddings(model=rag_cfg["embedding_model_name"])


# 全局变量（向后兼容）
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
