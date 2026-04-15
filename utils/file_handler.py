"""
文件读取相关函数
支持多种文件类型的加载
"""


import os
import hashlib
from utils.logger_handler import logger
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader, 
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
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

def listdir_with_allowed_types(directory: str, allowed_types: tuple[str], recursive: bool = True) -> list:
    """
    列出指定目录下的所有文件，过滤掉不允许的文件类型。
    :param recursive: 是否递归子目录
    """
    files = []
    if not os.path.isdir(directory):
        logger.error(f"[列出文件]路径{directory}不是一个目录")
        return files

    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(allowed_types):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(allowed_types):
                files.append(file_path)
    return files


def pdf_loader(file_path: str,password: str = None) -> list[Document]:
    """
    PDF文件加载器，返回文件内容
    """
    return PyPDFLoader(file_path,password=password).load()

def docx_loader(file_path: str) -> list[Document]:
    """
    docx文件加载器，返回文件内容。
    先尝试 UnstructuredWordDocumentLoader，若缺少 docx 依赖则降级为 docx2txt。
    """
    try:
        return UnstructuredWordDocumentLoader(file_path).load()
    except ModuleNotFoundError as e:
        logger.warning(f"[加载Word文件]缺少依赖: {e}. 尝试降级使用 docx2txt 加载。")
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            if text:
                return [Document(page_content=text, metadata={"source": file_path})]
            return []
        except ModuleNotFoundError:
            logger.error(
                "[加载Word文件]未安装 python-docx 或 docx2txt。请运行 pip install python-docx docx2txt"
            )
            return []
        except Exception as exc:
            logger.error(f"[加载Word文件]docx2txt 加载失败: {exc}")
            return []
    except Exception as exc:
        logger.error(f"[加载Word文件]加载文件{file_path}时发生错误：{exc}")
        return []

def txt_loader(file_path: str) -> list[Document]:
    """
    txt文件加载器，返回文件内容，支持常见编码自动回退
    """
    for encoding in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            return TextLoader(file_path, encoding=encoding).load()
        except Exception as e:
            logger.debug(f"[加载文本文件]编码 {encoding} 失败：{e}")
    logger.error(f"[加载文本文件]文件 {file_path} 无法读取，尝试过多种编码")
    return []

def md_loader(file_path: str) -> list[Document]:
    """
    Markdown文件加载器，返回文件内容
    """
    try:
        return UnstructuredMarkdownLoader(file_path).load()
    except Exception as e:
        logger.warning(f"[加载Markdown文件]加载文件{file_path}失败，尝试纯文本读取：{e}")
        return txt_loader(file_path)

def excel_loader(file_path: str) -> list[Document]:
    """
    Excel文件加载器（支持xlsx和xls），返回文件内容
    """
    try:
        return UnstructuredExcelLoader(file_path).load()
    except Exception as e:
        logger.error(f"[加载Excel文件]加载文件{file_path}时发生错误：{e}")
        return []

def csv_loader(file_path: str) -> list[Document]:
    """
    CSV文件加载器，返回文件内容
    """
    try:
        return CSVLoader(file_path, encoding='utf-8').load()
    except Exception as e:
        logger.error(f"[加载CSV文件]加载文件{file_path}时发生错误：{e}")
        return []

def html_loader(file_path: str) -> list[Document]:
    """
    HTML文件加载器，返回文件内容
    """
    try:
        return UnstructuredHTMLLoader(file_path).load()
    except Exception as e:
        logger.warning(f"[加载HTML文件]加载文件{file_path}失败，尝试纯文本读取：{e}")
        return txt_loader(file_path)

def get_file_loader(file_path: str):
    """
    根据文件扩展名返回对应的加载器函数
    :param file_path: 文件路径
    :return: 加载器函数或None
    """
    if file_path.endswith('.pdf'):
        return pdf_loader
    elif file_path.endswith('.docx'):
        return docx_loader
    elif file_path.endswith('.txt'):
        return txt_loader
    elif file_path.endswith('.md'):
        return md_loader
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return excel_loader
    elif file_path.endswith('.csv'):
        return csv_loader
    elif file_path.endswith('.html') or file_path.endswith('.htm'):
        return html_loader
    elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.gif'):
        return image_loader
    else:
        return None

def image_loader(file_path: str) -> list[Document]:
    """
    图片文件加载器，返回包含图片信息的Document
    :param file_path: 图片文件路径
    :return: Document对象列表
    """
    from rag.image_index import image_index
    
    image_id = image_index.add_image(file_path)
    image_index.save()
    
    if image_id:
        logger.info(f"[加载图片]图片{file_path}已添加到索引，ID: {image_id}")
        return [Document(
            page_content=f"图片文件: {os.path.basename(file_path)}, 图片ID: {image_id}",
            metadata={
                "source": file_path,
                "image_id": image_id,
                "file_type": "image"
            }
        )]
    else:
        logger.error(f"[加载图片]无法添加图片{file_path}到索引")
        return []

def load_file_content(file_path: str) -> list[Document]:
    """
    统一的文件加载函数，根据文件类型调用对应的加载器
    :param file_path: 文件路径
    :return: Document对象列表
    """
    loader = get_file_loader(file_path)
    if loader:
        return loader(file_path)
    else:
        logger.warning(f"[加载文件]不支持的文件类型：{file_path}")
        return []

# 支持的文件类型列表
SUPPORTED_FILE_TYPES = ('.txt', '.pdf', '.docx', '.md', '.xlsx', '.xls', '.csv', '.html', '.htm', '.png', '.jpg', '.jpeg', '.gif')