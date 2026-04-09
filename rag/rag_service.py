"""
rag总结服务：用户提问，搜索参考资料，将参考资料和用户问题一起输入模型，生成总结报告
"""
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document



class RagSummerizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_txt = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_txt)
        self.model = chat_model
        self.chain = self._init_chain()
    
    def _init_chain(self):
        chain = self.prompt_template | self.model |StrOutputParser()
        return chain
    
    def retrieve_docs(self, query: str) -> list[Document]:
        try:
            docs = self.retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"[RAG总结服务]检索相关文档时发生错误：{repr(e)},exc_info=True")
            raise e
    def rag_summarize(self, query: str) -> str:
        try:
            docs = self.retrieve_docs(query)

            context = ""
            count = 0
            for doc in docs:
                count += 1
                context += f"参考资料{count}：参考资料:{doc.page_content}|参考元数据：{doc.metadata}\n"
            
            summary = self.chain.invoke(
                {
                    "input": query,
                    "context": context
                }
            )
            return summary
        except Exception as e:
            logger.error(f"[RAG总结服务]生成总结报告时发生错误：{repr(e)},exc_info=True")
            raise e
if __name__ == "__main__":
    rag = RagSummerizeService()
    query = "小户型适合什么扫地机器人？"
    summary = rag.rag_summarize(query)
    print(summary)