from langchain_chroma import Chroma
from utils.config_handler import chroma_cfg
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_types, load_file_content, SUPPORTED_FILE_TYPES
import os
import shutil
import json
import re
from datetime import datetime
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

class VectorStoreService:
    def __init__(self, db_name: str = None):
        self.persist_base_dir = get_abs_path(chroma_cfg['persist_directory'])
        self.default_db_name = chroma_cfg.get('default_database_name', 'default')
        self.current_db = VectorStoreService._sanitize_db_name(db_name) if db_name else self._select_initial_database()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_cfg['chunk_size'], 
            chunk_overlap=chroma_cfg['chunk_overlap'],
            separators=chroma_cfg['separators'],
            length_function=len
            )
        self._init_store()
    
    def _select_initial_database(self) -> str:
        """优先使用已有数据库目录，否则使用配置默认数据库名。"""
        existing_dbs = self.list_databases()
        if not existing_dbs:
            return chroma_cfg['collection_name']
        if chroma_cfg['collection_name'] in existing_dbs:
            return chroma_cfg['collection_name']
        return existing_dbs[0]

    @staticmethod
    def _sanitize_db_name(name: str) -> str:
        """清理数据库名称，使其符合ChromaDB的要求：只包含[a-zA-Z0-9._-]，以字母或数字开头和结尾"""
        if not name:
            return name
        # 替换空格为下划线
        sanitized = name.replace(' ', '_')
        # 只保留允许的字符
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
        # 确保以字母或数字开头
        sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
        # 确保以字母或数字结尾
        sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
        # 如果为空，返回默认
        if not sanitized:
            sanitized = 'default'
        return sanitized

    def _init_store(self):
        """初始化向量存储"""
        persist_dir = os.path.join(self.persist_base_dir, self.current_db)
        os.makedirs(persist_dir, exist_ok=True)
        self.vectors_store = Chroma(
            collection_name=self.current_db,
            persist_directory=persist_dir,
            embedding_function=embed_model
        )
    
    def get_md5_store_path(self) -> str:
        """获取MD5存储文件路径"""
        return os.path.join(self.persist_base_dir, self.current_db, "md5.txt")

    def get_metadata_path(self) -> str:
        """获取文件元数据存储路径"""
        return os.path.join(self.persist_base_dir, self.current_db, "file_records.json")

    def load_file_metadata(self) -> dict:
        """加载当前数据库的文件元数据"""
        metadata_path = self.get_metadata_path()
        if not os.path.exists(metadata_path):
            return {}
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"[文件元数据]加载失败：{exc}")
            return {}

    def save_file_metadata(self, md5_hex: str, file_name: str, chunks: int) -> None:
        """保存文件元数据"""
        metadata = self.load_file_metadata()
        metadata[md5_hex] = {
            "md5": md5_hex,
            "file_name": file_name,
            "chunks": chunks,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(self.get_metadata_path(), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"[文件元数据]保存失败：{exc}")

    def get_retriever(self):
        return self.vectors_store.as_retriever(search_kwargs={"k": chroma_cfg['k']})

    def _is_ignored_category(self, name: str) -> bool:
        """判断目录名是否为系统级或 UUID 类别，应该屏蔽在页面中。"""
        if name.startswith('.') or name.startswith('_'):
            return True
        if re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', name):
            return True
        return False

    def get_data_categories(self) -> list:
        """获取 data_path 下的顶级目录作为数据分类源"""
        data_root = get_abs_path(chroma_cfg['data_path'])
        if not os.path.isdir(data_root):
            return []
        categories = []
        for entry in os.scandir(data_root):
            if entry.is_dir() and not self._is_ignored_category(entry.name):
                categories.append(entry.name)

        if self.get_data_files(self.default_db_name):
            categories.insert(0, self.default_db_name)
        return sorted(categories)

    def get_data_files(self, category: str | None = None) -> list:
        """获取 data_path 下指定分类或根目录的所有允许文件"""
        base_dir = get_abs_path(chroma_cfg['data_path'])
        if category == self.default_db_name or category is None:
            source_dir = base_dir
            recursive = False
        else:
            source_dir = os.path.join(base_dir, category)
            recursive = True
        if not os.path.isdir(source_dir):
            return []
        files = listdir_with_allowed_types(source_dir, tuple(chroma_cfg['allow_knowledge_file_type']), recursive=recursive)
        return sorted(files)

    def load_documents_from_path(self, source_path: str, recursive: bool = True) -> dict:
        """从指定目录加载文件到当前数据库"""
        if not os.path.isdir(source_path):
            return {"success": False, "message": f"目录不存在：{source_path}", "imported_files": 0}

        allowed_files = listdir_with_allowed_types(source_path, tuple(chroma_cfg['allow_knowledge_file_type']), recursive=recursive)
        if not allowed_files:
            return {"success": False, "message": "未找到可导入的文件。", "imported_files": 0}

        imported_count = 0
        for path in allowed_files:
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None or self.check_md5_hex(md5_hex):
                continue
            try:
                documents = self.get_file_documents(path)
                if not documents:
                    continue
                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    continue
                self.vectors_store.add_documents(split_documents)
                self.save_md5_hex(md5_hex)
                self.save_file_metadata(md5_hex, os.path.basename(path), len(split_documents))
                imported_count += 1
            except Exception as exc:
                logger.error(f"[数据导入]加载文件{path}时发生错误：{exc}")
                continue

        return {
            "success": True,
            "message": f"已导入 {imported_count} 个新文件到数据库 {self.current_db}",
            "imported_files": imported_count,
        }

    def import_data_category(self, category_name: str) -> dict:
        """将 data_path 下的某个分类文件夹导入到同名数据库"""
        data_root = get_abs_path(chroma_cfg['data_path'])
        if category_name == self.default_db_name:
            source_path = data_root
            recursive = False
        else:
            source_path = os.path.join(data_root, category_name)
            recursive = True

        if not os.path.isdir(source_path):
            return {"success": False, "message": f"数据目录不存在：{category_name}"}

        sanitized_db_name = VectorStoreService._sanitize_db_name(category_name)
        if sanitized_db_name not in self.list_databases():
            self.create_database(sanitized_db_name)

        target_store = VectorStoreService(db_name=sanitized_db_name)
        return target_store.load_documents_from_path(source_path, recursive=recursive)
    
    def check_md5_hex(self, md5_for_check: str) -> bool:
        """
        检查MD5值是否已经存在于向量库中
        """
        md5_file_path = self.get_md5_store_path()
        if not os.path.exists(md5_file_path):
            os.makedirs(os.path.dirname(md5_file_path), exist_ok=True)
            open(md5_file_path, 'w', encoding='utf-8').close()
            return False
        with open(md5_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip() == md5_for_check:
                    return True
            return False
            
    def save_md5_hex(self, md5_hex: str):
        """
        将MD5值保存到存储文件中
        """
        md5_file_path = self.get_md5_store_path()
        os.makedirs(os.path.dirname(md5_file_path), exist_ok=True)
        with open(md5_file_path, 'a', encoding='utf-8') as f:
            f.write(md5_hex + '\n')

    def get_file_documents(self, file_path: str) -> list:
        """
        根据文件类型调用不同的加载器加载文件内容，返回Document对象列表
        """
        return load_file_content(file_path)

    def load_document(self):
        """
        从数据文件夹加载文件，转为向量存入向量库
        计算文件的MD5值，判断文件是否已经被加载过，避免重复加载
        """
        allowed_file_pathes = listdir_with_allowed_types(get_abs_path(chroma_cfg['data_path']), tuple(chroma_cfg['allow_knowledge_file_type']))

        for path in allowed_file_pathes:
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None:
                continue
            if self.check_md5_hex(md5_hex):
                logger.info(f"[加载知识库文件]文件{path}已经被加载过，跳过")
                continue
            try:
                logger.info(f"[加载知识库文件]正在加载文件{path}，请稍候...")
                documents = self.get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库文件]文件{path}没有加载到内容，跳过")
                    continue
                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning(f"[加载知识库文件]文件{path}没有分割出内容，跳过")
                    continue
                self.vectors_store.add_documents(split_documents)
                self.save_md5_hex(md5_hex)
                logger.info(f"[加载知识库文件]成功加载文件{path}，并保存MD5值")
            except Exception as e:
                logger.error(f"[加载知识库文件]加载文件{path}时发生错误：{repr(e)},exc_info=True")
                continue

    def upload_file(self, file_path: str, file_name: str) -> dict:
        """
        上传单个文件到知识库
        :param file_path: 文件临时路径
        :param file_name: 原始文件名
        :return: 上传结果字典
        """
        try:
            md5_hex = get_file_md5_hex(file_path)
            if md5_hex is None:
                return {"success": False, "message": "无法计算文件MD5值"}
            
            if self.check_md5_hex(md5_hex):
                return {"success": False, "message": f"文件 {file_name} 已存在于知识库中"}

            logger.info(f"[上传文件]正在处理文件 {file_name}...")
            documents = self.get_file_documents(file_path)
            if not documents:
                return {"success": False, "message": f"无法读取文件 {file_name} 的内容"}
            
            split_documents = self.splitter.split_documents(documents)
            if not split_documents:
                return {"success": False, "message": f"文件 {file_name} 分割后没有内容"}
            
            self.vectors_store.add_documents(split_documents)
            self.save_md5_hex(md5_hex)
            self.save_file_metadata(md5_hex, file_name, len(split_documents))
            
            logger.info(f"[上传文件]成功上传文件 {file_name}")
            return {
                "success": True, 
                "message": f"文件 {file_name} 上传成功",
                "chunks": len(split_documents),
                "md5": md5_hex
            }
        except Exception as e:
            logger.error(f"[上传文件]上传文件 {file_name} 时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"上传失败：{str(e)}"}

    def get_uploaded_files(self) -> list:
        """
        获取已上传文件的元数据信息列表
        :return: 文件元数据列表
        """
        metadata = self.load_file_metadata()
        if metadata:
            return list(metadata.values())

        md5_list = []
        md5_file_path = self.get_md5_store_path()
        if os.path.exists(md5_file_path):
            with open(md5_file_path, 'r', encoding='utf-8') as f:
                md5_list = [line.strip() for line in f.readlines() if line.strip()]
        return [{"md5": md5, "file_name": "未知文件名", "chunks": 0, "uploaded_at": "未知"} for md5 in md5_list]

    def remove_file(self, md5_to_remove: str) -> dict:
        """
        从知识库中删除指定MD5的文件
        :param md5_to_remove: 要删除的文件MD5值
        :return: 删除结果字典
        """
        try:
            md5_file_path = self.get_md5_store_path()
            if not os.path.exists(md5_file_path):
                return {"success": False, "message": "MD5存储文件不存在"}

            with open(md5_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if md5_to_remove + '\n' not in lines and md5_to_remove not in [line.strip() for line in lines]:
                return {"success": False, "message": "该文件不存在于知识库中"}

            with open(md5_file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    if line.strip() != md5_to_remove:
                        f.write(line)

            metadata = self.load_file_metadata()
            if md5_to_remove in metadata:
                metadata.pop(md5_to_remove, None)
                try:
                    with open(self.get_metadata_path(), 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception as exc:
                    logger.error(f"[删除文件]更新元数据失败：{exc}")

            logger.info(f"[删除文件]成功删除MD5: {md5_to_remove}")
            return {"success": True, "message": "文件删除成功"}
        except Exception as e:
            logger.error(f"[删除文件]删除文件时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"删除失败：{str(e)}"}

    def reparse_file(self, file_path: str, file_name: str) -> dict:
        """
        重新解析文件（先删除旧的，再重新上传）
        :param file_path: 文件临时路径
        :param file_name: 原始文件名
        :return: 重新解析结果字典
        """
        try:
            md5_hex = get_file_md5_hex(file_path)
            if md5_hex is None:
                return {"success": False, "message": "无法计算文件MD5值"}

            if self.check_md5_hex(md5_hex):
                self.remove_file(md5_hex)

            result = self.upload_file(file_path, file_name)
            if result["success"]:
                result["message"] = f"文件 {file_name} 重新解析成功"
            return result
        except Exception as e:
            logger.error(f"[重新解析]重新解析文件 {file_name} 时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"重新解析失败：{str(e)}"}

    def delete_database(self) -> dict:
        """
        删除当前知识库数据库
        :return: 删除结果字典
        """
        try:
            persist_dir = os.path.join(self.persist_base_dir, self.current_db)
            md5_file_path = self.get_md5_store_path()

            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                logger.info(f"[删除数据库]成功删除向量库目录: {persist_dir}")

            logger.info(f"[删除数据库]成功删除数据库: {self.current_db}")
            return {"success": True, "message": f"数据库 {self.current_db} 删除成功"}
        except Exception as e:
            logger.error(f"[删除数据库]删除数据库时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"删除失败：{str(e)}"}

    def get_collection_stats(self) -> dict:
        """
        获取向量库统计信息
        :return: 统计信息字典
        """
        try:
            count = self.vectors_store._collection.count()
            uploaded_files = self.get_uploaded_files()
            return {
                "success": True,
                "total_chunks": count,
                "total_files": len(uploaded_files),
                "db_name": self.current_db
            }
        except Exception as e:
            logger.error(f"[获取统计]获取统计信息时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"获取统计失败：{str(e)}"}

    @staticmethod
    def _migrate_invalid_db_names():
        """迁移无效的数据库名称目录到有效的名称"""
        persist_base_dir = get_abs_path(chroma_cfg['persist_directory'])
        if not os.path.exists(persist_base_dir):
            return
        for item in os.listdir(persist_base_dir):
            item_path = os.path.join(persist_base_dir, item)
            if os.path.isdir(item_path):
                sanitized = VectorStoreService._sanitize_db_name(item)
                if sanitized != item:
                    new_path = os.path.join(persist_base_dir, sanitized)
                    if not os.path.exists(new_path):
                        try:
                            os.rename(item_path, new_path)
                            logger.info(f"迁移数据库目录: {item} -> {sanitized}")
                        except Exception as e:
                            logger.warning(f"迁移数据库目录失败 {item}: {e}，可能正在使用中，跳过")
                    else:
                        logger.warning(f"迁移目标已存在，跳过: {item} -> {sanitized}")

    @staticmethod
    def list_databases() -> list:
        """
        获取所有可用的数据库列表
        :return: 数据库名称列表
        """
        VectorStoreService._migrate_invalid_db_names()
        persist_base_dir = get_abs_path(chroma_cfg['persist_directory'])
        db_list = []
        if os.path.exists(persist_base_dir):
            for item in os.listdir(persist_base_dir):
                item_path = os.path.join(persist_base_dir, item)
                if os.path.isdir(item_path):
                    db_list.append(item)
        return sorted(db_list)
    
    def create_database(self, db_name: str) -> dict:
        """
        创建新数据库
        :param db_name: 数据库名称
        :return: 创建结果字典
        """
        try:
            if not db_name or not db_name.strip():
                return {"success": False, "message": "数据库名称不能为空"}
            
            db_name = VectorStoreService._sanitize_db_name(db_name.strip())
            
            if db_name in self.list_databases():
                return {"success": False, "message": f"数据库 {db_name} 已存在"}
            
            persist_dir = os.path.join(self.persist_base_dir, db_name)
            os.makedirs(persist_dir, exist_ok=True)
            
            md5_file_path = os.path.join(persist_dir, "md5.txt")
            open(md5_file_path, 'w', encoding='utf-8').close()
            
            logger.info(f"[创建数据库]成功创建数据库: {db_name}")
            return {"success": True, "message": f"数据库 {db_name} 创建成功"}
        except Exception as e:
            logger.error(f"[创建数据库]创建数据库时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"创建失败：{str(e)}"}
    
    def switch_database(self, db_name: str) -> dict:
        """
        切换到指定数据库
        :param db_name: 数据库名称
        :return: 切换结果字典
        """
        try:
            db_name = VectorStoreService._sanitize_db_name(db_name)
            
            if db_name == self.current_db:
                return {"success": True, "message": f"已在数据库 {db_name} 中"}
            
            db_list = self.list_databases()
            if db_name not in db_list:
                return {"success": False, "message": f"数据库 {db_name} 不存在"}
            
            self.current_db = db_name
            self._init_store()
            
            logger.info(f"[切换数据库]成功切换到数据库: {db_name}")
            return {"success": True, "message": f"成功切换到数据库 {db_name}"}
        except Exception as e:
            logger.error(f"[切换数据库]切换数据库时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"切换失败：{str(e)}"}

if __name__ == "__main__":
    vector_store_service = VectorStoreService()
    vector_store_service.load_document()
    retriever = vector_store_service.get_retriever()
    docs = retriever.invoke("机器学习是什么？")
    for doc in docs:
        print(doc.page_content)
        print("-"*20)