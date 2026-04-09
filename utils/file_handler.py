"""
文件读取相关函数
"""


import os
import hashlib
from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_core.documents import Document

def get_file_md5_hex(file_path: str) -> str:
    """
    获取文件的MD5值，返回16进制字符串
    """
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是一个文件")
        return
    md5_hash = hashlib.md5()

    chunk_size = 4096  # 4KB 分片读取文件，避免一次性加载大文件到内存
    try:
        with open(file_path, 'rb') as f:#以二进制方式打开文件
            while chunk := f.read(chunk_size):#循环读取文件内容，每次读取4KB
                md5_hash.update(chunk)#更新MD5对象的状态
        return md5_hash.hexdigest()#返回MD5值的16进制字符串表示
    except Exception as e:
        logger.error(f"[md5计算]计算文件{file_path}的MD5值时发生错误：{e}")
        return None

def listdir_with_allowed_types(directory: str, allowed_types: tuple[str]) -> list:
    """
    列出指定目录下的所有文件，过滤掉不允许的文件类型
    """
    files = []
    if not os.path.isdir(directory):
        logger.error(f"[列出文件]路径{directory}不是一个目录")
        return files

    for filename in os.listdir(directory):
        if filename.endswith(allowed_types):
            files.append(os.path.join(directory, filename))
        

    return tuple(files)

def pdf_loader(file_path: str,password: str = None) -> list[Document]:
    """
    PDF文件加载器，返回文件内容
    """
    return PyPDFLoader(file_path,password=password).load()

def docx_loader(file_path: str) -> list[Document]:
    """
    docx文件加载器，返回文件内容
    """
    return UnstructuredWordDocumentLoader(file_path).load()

def txt_loader(file_path: str) -> list[Document]:
    """
    txt文件加载器，返回文件内容
    """
    return TextLoader(file_path,encoding='utf-8').load()