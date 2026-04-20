"""
Ollama Provider - 本地Ollama实现

适合完全离线、隐私敏感、无API成本的场景。
"""
from typing import List, Dict, Any, Optional
from model.base import BaseLLM, BaseEmbedding
from utils.logger_handler import logger


class OllamaLLM(BaseLLM):
    """Ollama API实现 - 使用LangChain ChatOllama确保完全兼容性"""
    
    def __init__(self, model: str = "qwen3.5:9b", base_url: str = "http://localhost:11434"):
        from langchain_ollama import ChatOllama
        self._model_name = model
        self._base_url = base_url
        self._client = ChatOllama(model=model, base_url=base_url)
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """核心聊天接口"""
        try:
            response = self._client.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"[OllamaLLM] 聊天请求失败: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """简单文本生成接口"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)
    
    def chat_stream(self, messages: List[Dict[str, str]]):
        """流式聊天接口"""
        try:
            for chunk in self._client.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"[OllamaLLM] 流式请求失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """获取token数量"""
        return len(text) // 4
    
    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self._model_name
    
    # ========== 直接委托给底层ChatOllama客户端 ==========
    
    def bind_tools(self, tools, **kwargs):
        """绑定工具到模型（兼容 LangChain Agent）"""
        return self._client.bind_tools(tools, **kwargs)
    
    def bind(self, **kwargs):
        """绑定参数到模型"""
        return self._client.bind(**kwargs)
    
    def invoke(self, input, **kwargs):
        """调用模型（兼容 LangChain Runnable）"""
        return self._client.invoke(input, **kwargs)
    
    def stream(self, input, **kwargs):
        """流式调用模型（兼容 LangChain Runnable）"""
        return self._client.stream(input, **kwargs)
    
    @property
    def model(self):
        """兼容旧代码访问 model 属性"""
        return self._model_name
    
    def __getattr__(self, name):
        """
        委托未定义的属性和方法给底层客户端
        这确保了与 LangChain 的完全兼容性
        """
        if hasattr(self._client, name):
            return getattr(self._client, name)
        raise AttributeError(f"'OllamaLLM' object has no attribute '{name}'")


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding实现"""
    
    def __init__(self, model: str = "Mxbai-embed-large", base_url: str = "http://localhost:11434"):
        from langchain_ollama import OllamaEmbeddings
        self._model_name = model
        self._base_url = base_url
        self._client = OllamaEmbeddings(model=model, base_url=base_url)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.error(f"[OllamaEmbedding] 向量化失败: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化（兼容Chroma接口）"""
        return self.embed(texts)
    
    def embed_single(self, text: str) -> List[float]:
        """单文本向量化"""
        return self._client.embed_query(text)
    
    def embed_query(self, text: str) -> List[float]:
        """单文本向量化（兼容LangChain接口）"""
        return self.embed_single(text)
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return 1024  # Mxbai-embed-large默认维度