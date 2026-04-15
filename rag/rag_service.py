"""
rag总结服务：用户提问，搜索参考资料，返回检索到的文档内容
"""
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from langchain_core.documents import Document


class RagSummerizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
    
    def retrieve_docs(self, query: str) -> list[Document]:
        try:
            docs = self.retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"[RAG总结服务]检索相关文档时发生错误：{repr(e)},exc_info=True")
            raise e
    
    def rag_summarize(self, query: str) -> str:
        """
        检索相关文档并返回格式化的文档内容，由agent负责生成最终总结
        """
        try:
            docs = self.retrieve_docs(query)

            context = ""
            count = 0
            for doc in docs:
                count += 1
                context += f"参考资料{count}：{doc.page_content}\n"
            
            if not context:
                return "未检索到相关资料"
            
            return context
        except Exception as e:
            logger.error(f"[RAG总结服务]检索文档时发生错误：{repr(e)},exc_info=True")
            raise e

if __name__ == "__main__":
    rag = RagSummerizeService()
    query = "小户型适合什么扫地机器人？"
    result = rag.rag_summarize(query)
    print(result)