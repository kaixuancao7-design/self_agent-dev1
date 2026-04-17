"""
Tongyi (阿里云通义) Provider
"""
from typing import List, Dict, Any, Optional, Generator
from model.base import BaseLLM
from utils.logger_handler import logger


class TongyiLLM(BaseLLM):
    """
    阿里云通义大模型 Provider
    """
    
    def __init__(self, model: str = "qwen3-max"):
        """
        初始化
        
        :param model: 模型名称
        """
        try:
            from langchain_community.chat_models.tongyi import ChatTongyi
            self._client = ChatTongyi(model=model)
            self._model_name = model
            logger.info(f"[TongyiLLM] 初始化成功: {model}")
        except ImportError as e:
            logger.error(f"[TongyiLLM] 导入失败: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        同步聊天接口
        
        :param messages: 消息列表
        :return: 回复内容
        """
        try:
            response = self._client.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"[TongyiLLM] 聊天调用失败: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        简单文本生成接口
        
        :param prompt: 输入提示词
        :return: 生成的文本
        """
        return self.chat([{"role": "user", "content": prompt}])
    
    def chat_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        流式聊天接口
        
        :param messages: 消息列表
        :yield: 内容片段
        """
        try:
            stream = self._client.stream(messages)
            for chunk in stream:
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, dict) and 'content' in chunk:
                    yield chunk['content']
        except Exception as e:
            logger.error(f"[TongyiLLM] 流式聊天调用失败: {e}")
            raise
    
    def get_token_count(self, text: str) -> int:
        """
        获取文本token数量（估算）
        
        :param text: 输入文本
        :return: token数量
        """
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - chinese_chars - english_chars
        return chinese_chars + int(english_chars * 0.25) + other_chars
    
    @property
    def model_name(self) -> str:
        """
        获取模型名称
        
        :return: 模型名称
        """
        return self._model_name
    
    # ============ LangChain Agent 兼容方法 ============
    
    def bind_tools(self, tools, **kwargs):
        """
        绑定工具到模型（兼容 LangChain Agent）
        
        :param tools: 工具列表
        :param kwargs: 其他参数
        :return: 绑定工具后的模型
        """
        return self._client.bind_tools(tools, **kwargs)
    
    def bind(self, **kwargs):
        """
        绑定参数到模型
        
        :param kwargs: 参数
        :return: 绑定后的模型
        """
        return self._client.bind(**kwargs)
    
    def invoke(self, input, **kwargs):
        """
        调用模型（兼容 LangChain Runnable）
        
        :param input: 输入
        :param kwargs: 其他参数
        :return: 响应
        """
        return self._client.invoke(input, **kwargs)
    
    def stream(self, input, **kwargs):
        """
        流式调用模型（兼容 LangChain Runnable）
        
        :param input: 输入
        :param kwargs: 其他参数
        :yield: 响应片段
        """
        return self._client.stream(input, **kwargs)
    
    async def ainvoke(self, input, **kwargs):
        """
        异步调用模型（兼容 LangChain Runnable）
        
        :param input: 输入
        :param kwargs: 其他参数
        :return: 响应
        """
        return await self._client.ainvoke(input, **kwargs)
    
    async def astream(self, input, **kwargs):
        """
        异步流式调用模型（兼容 LangChain Runnable）
        
        :param input: 输入
        :param kwargs: 其他参数
        :yield: 响应片段
        """
        async for chunk in self._client.astream(input, **kwargs):
            yield chunk
    
    # ============ 委托属性 ============
    
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
        raise AttributeError(f"'TongyiLLM' object has no attribute '{name}'")
