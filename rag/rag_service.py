"""
rag总结服务：用户提问，搜索参考资料，返回检索到的文档内容
支持重排功能提升检索效果
支持查询缓存机制
"""
from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from utils.logger_handler import logger
from langchain_core.documents import Document
from utils.config_handler import rag_cfg
from utils.cache_service import cache_service


class RagSummerizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        # 初始化混合检索器（支持重排）
        self._init_hybrid_retriever()
        # 是否启用缓存
        self.cache_enabled = rag_cfg.get("enable_cache", True)
    
    def _init_hybrid_retriever(self):
        """初始化混合检索器"""
        try:
            rerank_method = rag_cfg.get("rerank_method", "linear")
            self.hybrid_retriever = get_hybrid_retriever(self.vector_store)
            logger.info(f"[RAG总结服务] 初始化混合检索器，重排方法: {rerank_method}")
        except Exception as e:
            logger.warning(f"[RAG总结服务] 初始化混合检索器失败，使用默认检索器: {e}")
            self.hybrid_retriever = None
    
    def retrieve_docs(self, query: str, use_hybrid: bool = True) -> list[Document]:
        """
        检索相关文档（带缓存）
        
        :param query: 用户查询
        :param use_hybrid: 是否使用混合检索（含重排）
        :return: 文档列表
        """
        # 尝试从缓存获取
        if self.cache_enabled:
            cache_key = f"retrieve_docs_{use_hybrid}_{rag_cfg.get('top_k', 5)}_{rag_cfg.get('rerank_method', 'linear')}"
            cached_docs = cache_service.get_query_result(query, key=cache_key)
            if cached_docs is not None:
                logger.info(f"[RAG总结服务] 检索缓存命中: {query[:30]}...")
                return cached_docs
        
        try:
            if use_hybrid and self.hybrid_retriever:
                # 使用混合检索器（含重排）
                rerank_method = rag_cfg.get("rerank_method", "linear")
                results = self.hybrid_retriever.retrieve(query, 
                                                       top_k=rag_cfg.get("top_k", 5), 
                                                       rerank_method=rerank_method)
                # 转换为 Document 对象
                docs = []
                for doc_id, info in results:
                    doc = Document(
                        page_content=info.get('content', ''),
                        metadata={'source': doc_id, 'score': info.get('score', 0.0)}
                    )
                    docs.append(doc)
                # 缓存结果
                if self.cache_enabled:
                    cache_service.set_query_result(query, docs, key=cache_key)
                return docs
            else:
                # 使用默认检索器
                docs = self.retriever.invoke(query)
                # 缓存结果
                if self.cache_enabled:
                    cache_service.set_query_result(query, docs, key=cache_key)
                return docs
        except Exception as e:
            logger.error(f"[RAG总结服务]检索相关文档时发生错误：{repr(e)},exc_info=True")
            raise e
    
    def rag_summarize(self, query: str, use_hybrid: bool = True) -> str:
        """
        检索相关文档并返回格式化的文档内容，由agent负责生成最终总结（带缓存）
        
        :param query: 用户查询
        :param use_hybrid: 是否使用混合检索（含重排）
        """
        # 尝试从缓存获取
        if self.cache_enabled:
            cache_key = f"rag_summarize_{use_hybrid}_{rag_cfg.get('top_k', 5)}_{rag_cfg.get('rerank_method', 'linear')}"
            cached_result = cache_service.get_query_result(query, key=cache_key)
            if cached_result is not None:
                logger.info(f"[RAG总结服务] 总结缓存命中: {query[:30]}...")
                return cached_result
        
        try:
            docs = self.retrieve_docs(query, use_hybrid=use_hybrid)

            context = ""
            count = 0
            for doc in docs:
                count += 1
                score = doc.metadata.get('score', 0.0)
                context += f"参考资料{count}（相关性: {score:.2f}）：{doc.page_content}\n"
            
            if not context:
                result = "未检索到相关资料"
            else:
                result = context
            
            # 缓存结果
            if self.cache_enabled:
                cache_service.set_query_result(query, result, key=cache_key)
            
            return result
        except Exception as e:
            logger.error(f"[RAG总结服务]检索文档时发生错误：{repr(e)},exc_info=True")
            raise e

if __name__ == "__main__":
    rag = RagSummerizeService()
    query = "小户型适合什么扫地机器人？"
    result = rag.rag_summarize(query)
    print(result)