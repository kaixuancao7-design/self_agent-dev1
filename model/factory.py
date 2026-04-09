from abc import ABC, abstractmethod
from typing import Optional
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import sys
# print("当前 Python 路径：")
# for path in sys.path:
#     print(path)
from utils.config_handler import rag_cfg, chroma_cfg, prompts_cfg, agent_cfg
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        return ChatTongyi(model=rag_cfg["chat_model_name"])
    
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        return DashScopeEmbeddings(model=rag_cfg["embedding_model_name"])
    


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()