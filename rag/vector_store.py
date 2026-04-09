from langchain_chroma import Chroma
from utils.config_handler import chroma_cfg
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_types, pdf_loader, docx_loader, txt_loader
import os
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self):
        self.vectors_store = Chroma(
            collection_name = chroma_cfg['collection_name'],
            persist_directory = chroma_cfg['persist_directory'],
            embedding_function = embed_model
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_cfg['chunk_size'], 
            chunk_overlap=chroma_cfg['chunk_overlap'],
            separators=chroma_cfg['separators'],
            length_function=len
            )

    def get_retriever(self):
        return self.vectors_store.as_retriever(search_kwargs={"k": chroma_cfg['k']})
    
    def load_document(self):
        """
        从数据文件夹加载文件，转为向量存入向量库
        计算文件的MD5值，判断文件是否已经被加载过，避免重复加载
        """
        def check_md5_hex(md5_for_check: str) -> bool:
            """
            检查MD5值是否已经存在于向量库中
            """
            if not os.path.exists(get_abs_path(chroma_cfg["md5_hex_store"])):
                open(get_abs_path(chroma_cfg["md5_hex_store"]), 'w',encoding='utf-8').close()#如果MD5值存储文件不存在，创建一个空文件
                return False #如果MD5值存储文件不存在，说明没有文件被加载过，直接返回False
            with open(get_abs_path(chroma_cfg["md5_hex_store"]), 'r',encoding='utf-8') as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True #如果MD5值已经存在于存储文件中，说明文件已经被加载过，返回True
                return False #如果MD5值不存在于存储文件中，返回False
            
        def save_md5_hex(md5_hex: str):
            """
            将MD5值保存到存储文件中
            """
            with open(get_abs_path(chroma_cfg["md5_hex_store"]), 'a',encoding='utf-8') as f:
                f.write(md5_hex + '\n')#将MD5值写入存储文件，每个MD5值占一行

        def get_file_documents(file_path: str) -> list:
            """
            根据文件类型调用不同的加载器加载文件内容，返回Document对象列表
            """
            if file_path.endswith('.pdf'):
                return pdf_loader(file_path)
            elif file_path.endswith('.docx'):
                return docx_loader(file_path)
            elif file_path.endswith('.txt'):
                return txt_loader(file_path)
            else:
                logger.warning(f"[加载文件]不支持的文件类型：{file_path}")
                return []
            
        allowed_file_pathes = listdir_with_allowed_types(get_abs_path(chroma_cfg['data_path']), tuple(chroma_cfg['allow_knowledge_file_type']))

        for path in allowed_file_pathes:
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None:
                continue #如果计算MD5值失败，跳过该文件
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库文件]文件{path}已经被加载过，跳过")
                continue #如果文件已经被加载过，跳过该文件
            try:
                logger.info(f"[加载知识库文件]正在加载文件{path}，请稍候...")
                documents = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库文件]文件{path}没有加载到内容，跳过")
                    continue #如果没有加载到内容，跳过该文件
                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[加载知识库文件]文件{path}没有分割出内容，跳过")
                    continue #如果没有分割出内容，跳过该文件
                #将内容存入向量库
                self.vectors_store.add_documents(split_documents)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库文件]成功加载文件{path}，并保存MD5值")
            except Exception as e:
                logger.error(f"[加载知识库文件]加载文件{path}时发生错误：{repr(e)},exc_info=True")
                continue #如果加载文件时发生错误，跳过该文件

if __name__ == "__main__":
    vector_store_service = VectorStoreService()
    vector_store_service.load_document()
    retriever = vector_store_service.get_retriever()
    docs = retriever.invoke("机器学习是什么？")
    for doc in docs:
        print(doc.page_content)
        print("-"*20)