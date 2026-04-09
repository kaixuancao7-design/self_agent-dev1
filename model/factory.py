from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_cfg, chroma_cfg, prompts_cfg, agent_cfg
from utils.logger_handler import logger
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass

class CloudChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        model_name = rag_cfg.get("cloud_chat_model_name", "qwen3-max")
        logger.info(f"[ModelFactory] 初始化云端模型：{model_name}")
        return ChatTongyi(model=model_name)


class OllamaChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        try:
            from langchain_community.chat_models.ollama import ChatOllama
        except Exception as exc:
            logger.error(f"[ModelFactory] 无法加载 Ollama 模型库：{exc}")
            raise

        model_name = agent_cfg.get("ollama_model_name", "llama2")
        base_url = agent_cfg.get("ollama_base_url", "http://localhost:11434")
        logger.info(f"[ModelFactory] 初始化本地 Ollama 模型：{model_name} @ {base_url}")
        return ChatOllama(model=model_name, base_url=base_url)


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
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


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=rag_cfg["embedding_model_name"])
    


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()