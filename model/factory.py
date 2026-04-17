"""
统一模型工厂 - 基于配置自动选择Provider

支持的Provider类型:
- azure: Azure OpenAI
- openai: OpenAI原生
- deepseek: DeepSeek
- ollama: Ollama本地部署
- tongyi: 阿里云通义千问
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
        
        :param provider: Provider类型 (azure/openai/deepseek/ollama/tongyi)
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
        elif provider == "tongyi":
            return LLMFactory._create_tongyi(**kwargs)
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
    
    @staticmethod
    def _create_tongyi(model: str = "qwen3-max", **kwargs) -> BaseLLM:
        """创建Tongyi实例"""
        from model.providers.tongyi_provider import TongyiLLM
        logger.info(f"[LLMFactory] 创建Tongyi实例: {model}")
        return TongyiLLM(model=model)


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
    def _create_ollama(model: str = "Mxbai-embed-large", base_url: str = "http://localhost:11434", **kwargs) -> BaseEmbedding:
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


# 全局变量 - 使用统一工厂类创建实例
def _create_chat_model() -> BaseLLM:
    """创建聊天模型实例"""
    llm_config = agent_cfg.get("llm", {})
    provider = llm_config.get("provider", "azure")
    model = llm_config.get("model", "gpt-4o")
    
    try:
        if provider == "ollama":
            base_url = llm_config.get("base_url", "http://localhost:11434")
            return LLMFactory.create(provider, model=model, base_url=base_url)
        else:
            api_key = llm_config.get("api_key", "")
            endpoint = llm_config.get("azure_endpoint", "")
            deployment_name = llm_config.get("azure_deployment_name", model)
            return LLMFactory.create(provider, api_key=api_key, endpoint=endpoint, 
                                    deployment_name=deployment_name, model=model)
    except Exception as e:
        logger.error(f"[LLMFactory] 创建聊天模型失败: {e}")
        raise


def _create_embed_model() -> BaseEmbedding:
    """创建嵌入模型实例"""
    embedding_config = rag_cfg.get("embedding", {})
    provider = embedding_config.get("provider", "openai")
    model = embedding_config.get("model", "text-embedding-3-small")
    
    try:
        if provider == "ollama":
            base_url = embedding_config.get("base_url", "http://localhost:11434")
            return EmbeddingFactory.create(provider, model=model, base_url=base_url)
        else:
            api_key = embedding_config.get("api_key", "")
            endpoint = embedding_config.get("azure_endpoint", "")
            deployment_name = embedding_config.get("azure_deployment_name", model)
            return EmbeddingFactory.create(provider, api_key=api_key, endpoint=endpoint,
                                          deployment_name=deployment_name, model=model)
    except Exception as e:
        logger.error(f"[EmbeddingFactory] 创建嵌入模型失败: {e}")
        raise


# 全局变量
chat_model = _create_chat_model()
embed_model = _create_embed_model()
