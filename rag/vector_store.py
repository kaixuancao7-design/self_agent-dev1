from langchain_chroma import Chroma
from utils.config_handler import chroma_cfg, rag_cfg
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_types, load_file_content, SUPPORTED_FILE_TYPES
import os
import shutil
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# 缓存管理器
class ChromaCacheManager:
    """Chroma缓存管理器"""
    
    def __init__(self, persist_base_dir: str):
        self.persist_base_dir = persist_base_dir
        self.cache_dir = os.path.join(persist_base_dir, "_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.query_cache = {}
        self._load_cache()
    
    def _get_cache_path(self, db_name: str) -> str:
        return os.path.join(self.cache_dir, f"{db_name}_cache.json")
    
    def _load_cache(self):
        """加载缓存数据"""
        try:
            for item in os.listdir(self.cache_dir):
                if item.endswith("_cache.json"):
                    db_name = item.replace("_cache.json", "")
                    cache_path = self._get_cache_path(db_name)
                    if os.path.exists(cache_path):
                        try:
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                self.query_cache[db_name] = json.load(f)
                        except Exception as e:
                            logger.warning(f"[缓存加载]加载缓存失败 {db_name}: {e}")
        except Exception as e:
            logger.warning(f"[缓存加载]初始化缓存失败: {e}")
    
    def _save_cache(self, db_name: str):
        """保存缓存数据"""
        try:
            cache_path = self._get_cache_path(db_name)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.query_cache.get(db_name, {}), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[缓存保存]保存缓存失败 {db_name}: {e}")
    
    def get_cache(self, db_name: str, query: str) -> Optional[List[Any]]:
        """获取查询缓存"""
        cache = self.query_cache.get(db_name, {})
        item = cache.get(query)
        if item:
            # 检查是否过期（TTL 5分钟）
            if time.time() - item.get('timestamp', 0) < 300:
                return item.get('results')
            else:
                # 过期则删除
                del cache[query]
                self._save_cache(db_name)
        return None
    
    def set_cache(self, db_name: str, query: str, results: List[Any]):
        """设置查询缓存"""
        if db_name not in self.query_cache:
            self.query_cache[db_name] = {}
        
        # 限制缓存大小（最多100条）
        cache = self.query_cache[db_name]
        if len(cache) >= 100:
            # 删除最旧的缓存
            oldest_key = min(cache.keys(), key=lambda k: cache[k].get('timestamp', 0))
            del cache[oldest_key]
        
        cache[query] = {
            'timestamp': time.time(),
            'results': results
        }
        self._save_cache(db_name)
    
    def clear_cache(self, db_name: str = None):
        """清除缓存"""
        if db_name:
            self.query_cache[db_name] = {}
            cache_path = self._get_cache_path(db_name)
            if os.path.exists(cache_path):
                os.remove(cache_path)
        else:
            self.query_cache = {}
            for item in os.listdir(self.cache_dir):
                if item.endswith("_cache.json"):
                    os.remove(os.path.join(self.cache_dir, item))


class VectorStoreService:
    # 全局缓存管理器
    _cache_manager = None
    
    def __init__(self, db_name: str = None):
        self.persist_base_dir = get_abs_path(chroma_cfg['persist_directory'])
        self.default_db_name = chroma_cfg.get('default_database_name', 'default')
        self.current_db = VectorStoreService._sanitize_db_name(db_name) if db_name else self._select_initial_database()
        
        # 初始化缓存管理器
        if VectorStoreService._cache_manager is None:
            VectorStoreService._cache_manager = ChromaCacheManager(self.persist_base_dir)
        self.cache_manager = VectorStoreService._cache_manager
        
        # 清理分隔符配置，移除空字符串以防止切分产生空chunk
        separators = chroma_cfg['separators']
        if "" in separators:
            separators = [s for s in separators if s != ""]
            logger.warning(f"[配置检查] 移除了分隔符列表中的空字符串")
        
        # 根据文件大小动态调整分块参数
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_cfg['chunk_size'], 
            chunk_overlap=chroma_cfg['chunk_overlap'],
            separators=separators,
            length_function=len
        )
        self._init_store()
    
    def _select_initial_database(self) -> str:
        """优先使用已有数据库目录，否则使用配置默认数据库名。"""
        existing_dbs = self.list_databases()
        valid_dbs = [db for db in existing_dbs if len(db) >= 3]
        if not valid_dbs:
            return chroma_cfg['collection_name']
        if chroma_cfg['collection_name'] in valid_dbs:
            return chroma_cfg['collection_name']
        return valid_dbs[0]

    @staticmethod
    def _sanitize_db_name(name: str) -> str:
        """清理数据库名称，使其符合ChromaDB的要求"""
        if not name:
            return 'default'
        sanitized = name.replace(' ', '_')
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)
        sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
        sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
        if not sanitized:
            sanitized = 'default'
        min_length, max_length = 3, 512
        if len(sanitized) < min_length:
            suffix = '_db' if len(sanitized) == 1 else '_d'
            sanitized = sanitized + suffix
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
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
    
    def _close_store(self):
        """关闭向量存储连接，释放文件句柄"""
        if hasattr(self, 'vectors_store') and self.vectors_store is not None:
            try:
                if hasattr(self.vectors_store, 'delete_collection'):
                    try:
                        self.vectors_store.delete_collection()
                    except Exception as e:
                        logger.warning(f"[关闭数据库]删除collection失败（可能已不存在）: {e}")
                
                if hasattr(self.vectors_store, '_client'):
                    client = self.vectors_store._client
                    if hasattr(client, 'close'):
                        client.close()
                
                self.vectors_store = None
                logger.info(f"[关闭数据库]成功关闭数据库连接: {self.current_db}")
            except Exception as e:
                logger.warning(f"[关闭数据库]关闭连接时发生警告: {e}")
    
    def _get_splitter_for_file(self, file_size: int) -> RecursiveCharacterTextSplitter:
        """根据文件大小获取合适的分块器"""
        # 默认配置
        chunk_size = chroma_cfg['chunk_size']
        chunk_overlap = chroma_cfg['chunk_overlap']
        
        # 大文件优化：文件越大，chunk越大，重叠越小
        if file_size > 10 * 1024 * 1024:  # >10MB
            chunk_size = min(chunk_size * 2, 2048)
            chunk_overlap = max(chunk_overlap // 2, 32)
            logger.info(f"[文件分块]大文件优化: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        elif file_size > 1 * 1024 * 1024:  # >1MB
            chunk_size = min(chunk_size * 1.5, 1024)
            chunk_overlap = max(chunk_overlap // 1.5, 48)
            logger.info(f"[文件分块]中文件优化: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        return RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            separators=chroma_cfg['separators'],
            length_function=len
        )

    def get_md5_store_path(self) -> str:
        return os.path.join(self.persist_base_dir, self.current_db, "md5.txt")

    def get_metadata_path(self) -> str:
        return os.path.join(self.persist_base_dir, self.current_db, "file_records.json")

    def load_file_metadata(self) -> dict:
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
        return self.vectors_store.as_retriever(search_kwargs={"k": rag_cfg['retrieval']['top_k']})

    def _is_ignored_category(self, name: str) -> bool:
        if name.startswith('.') or name.startswith('_'):
            return True
        if re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', name):
            return True
        return False

    def get_data_categories(self) -> list:
        data_root = get_abs_path(chroma_cfg['data_path'])
        os.makedirs(data_root, exist_ok=True)
        dbs = self.list_databases()
        dbs = list(dict.fromkeys(dbs))
        for db_name in dbs:
            db_data_dir = os.path.join(data_root, db_name)
            os.makedirs(db_data_dir, exist_ok=True)
        return sorted(dbs)

    def get_data_files(self, category: str | None = None) -> list:
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
                
                # 根据文件大小选择合适的分块器
                file_size = os.path.getsize(path)
                splitter = self._get_splitter_for_file(file_size)
                split_documents = splitter.split_documents(documents)
                
                # 过滤空的或过短的chunk
                split_documents = self._filter_empty_chunks(split_documents)
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
        md5_file_path = self.get_md5_store_path()
        os.makedirs(os.path.dirname(md5_file_path), exist_ok=True)
        with open(md5_file_path, 'a', encoding='utf-8') as f:
            f.write(md5_hex + '\n')

    def get_file_documents(self, file_path: str) -> list:
        return load_file_content(file_path)
    
    def _filter_empty_chunks(self, documents: list) -> list:
        """过滤空的或过短的文档chunk"""
        filtered = []
        min_length = 10
        
        for doc in documents:
            content = doc.page_content.strip() if doc.page_content else ""
            if len(content) >= min_length:
                filtered.append(doc)
            else:
                logger.warning(f"[Chunk过滤] 跳过空或过短的chunk (长度: {len(content)})")
        
        return filtered

    def load_document(self):
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
                
                file_size = os.path.getsize(path)
                splitter = self._get_splitter_for_file(file_size)
                split_documents = splitter.split_documents(documents)
                
                split_documents = self._filter_empty_chunks(split_documents)
                if not split_documents:
                    logger.warning(f"[加载知识库文件]文件{path}没有分割出有效内容，跳过")
                    continue
                
                self.vectors_store.add_documents(split_documents)
                self.save_md5_hex(md5_hex)
                logger.info(f"[加载知识库文件]成功加载文件{path}，并保存MD5值")
            except Exception as e:
                logger.error(f"[加载知识库文件]加载文件{path}时发生错误：{repr(e)},exc_info=True")
                continue

    def upload_file(self, file_path: str, file_name: str) -> dict:
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
            
            # 根据文件大小选择合适的分块器
            file_size = os.path.getsize(file_path)
            splitter = self._get_splitter_for_file(file_size)
            split_documents = splitter.split_documents(documents)
            
            # 过滤空的或过短的chunk
            split_documents = self._filter_empty_chunks(split_documents)
            if not split_documents:
                return {"success": False, "message": f"文件 {file_name} 分割后没有有效内容"}
            
            self.vectors_store.add_documents(split_documents)
            self.save_md5_hex(md5_hex)
            self.save_file_metadata(md5_hex, file_name, len(split_documents))
            
            data_dir = os.path.join(get_abs_path(chroma_cfg['data_path']), self.current_db)
            os.makedirs(data_dir, exist_ok=True)
            dest_file_path = os.path.join(data_dir, file_name)
            shutil.copy2(file_path, dest_file_path)
            logger.info(f"[上传文件]文件已保存到本地数据目录: {dest_file_path}")
            
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
            file_name = None
            if md5_to_remove in metadata:
                file_name = metadata[md5_to_remove].get('file_name')
                metadata.pop(md5_to_remove, None)
                try:
                    with open(self.get_metadata_path(), 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception as exc:
                    logger.error(f"[删除文件]更新元数据失败：{exc}")

            if file_name:
                data_dir = os.path.join(get_abs_path(chroma_cfg['data_path']), self.current_db)
                local_file_path = os.path.join(data_dir, file_name)
                if os.path.exists(local_file_path):
                    os.remove(local_file_path)
                    logger.info(f"[删除文件]已删除本地数据目录中的文件: {local_file_path}")

            logger.info(f"[删除文件]成功删除MD5: {md5_to_remove}")
            return {"success": True, "message": "文件删除成功"}
        except Exception as e:
            logger.error(f"[删除文件]删除文件时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"删除失败：{str(e)}"}

    def reparse_file(self, file_path: str, file_name: str) -> dict:
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
        import time
        
        try:
            persist_dir = os.path.join(self.persist_base_dir, self.current_db)
            data_dir = os.path.join(get_abs_path(chroma_cfg['data_path']), self.current_db)

            self._close_store()
            time.sleep(0.5)
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    if os.path.exists(persist_dir):
                        shutil.rmtree(persist_dir)
                        logger.info(f"[删除数据库]成功删除向量库目录: {persist_dir}")
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"[删除数据库]第 {attempt+1} 次删除失败，文件可能被占用，重试中...")
                        time.sleep(retry_delay)
                    else:
                        raise e

            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
                logger.info(f"[删除数据库]成功删除本地数据目录: {data_dir}")

            # 清除缓存
            self.cache_manager.clear_cache(self.current_db)

            logger.info(f"[删除数据库]成功删除数据库: {self.current_db}")
            return {"success": True, "message": f"数据库 {self.current_db} 删除成功"}
        except Exception as e:
            logger.error(f"[删除数据库]删除数据库时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"删除失败：{str(e)}"}

    def get_collection_stats(self) -> dict:
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
        VectorStoreService._migrate_invalid_db_names()
        persist_base_dir = get_abs_path(chroma_cfg['persist_directory'])
        db_list = []
        if os.path.exists(persist_base_dir):
            for item in os.listdir(persist_base_dir):
                item_path = os.path.join(persist_base_dir, item)
                if os.path.isdir(item_path):
                    db_list.append(item)
        db_list = list(dict.fromkeys(db_list))
        return sorted(db_list)
    
    def create_database(self, db_name: str) -> dict:
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
            
            data_dir = os.path.join(get_abs_path(chroma_cfg['data_path']), db_name)
            os.makedirs(data_dir, exist_ok=True)
            
            logger.info(f"[创建数据库]成功创建数据库: {db_name}，向量库目录: {persist_dir}，数据目录: {data_dir}")
            return {"success": True, "message": f"数据库 {db_name} 创建成功"}
        except Exception as e:
            logger.error(f"[创建数据库]创建数据库时发生错误：{repr(e)},exc_info=True")
            return {"success": False, "message": f"创建失败：{str(e)}"}
    
    def switch_database(self, db_name: str) -> dict:
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

    def retrieve_with_cache(self, query: str, top_k: int = None) -> List[Any]:
        """带缓存的检索方法"""
        top_k = top_k or rag_cfg['retrieval']['top_k']
        
        # 尝试从缓存获取
        cached_results = self.cache_manager.get_cache(self.current_db, query)
        if cached_results:
            logger.info(f"[缓存命中]查询: {query}")
            return cached_results
        
        # 执行实际检索
        try:
            retriever = self.get_retriever()
            results = retriever.invoke(query)
            
            # 缓存结果
            self.cache_manager.set_cache(self.current_db, query, results)
            
            return results
        except Exception as e:
            logger.error(f"[检索失败]查询失败: {e}")
            return []


if __name__ == "__main__":
    vector_store_service = VectorStoreService()
    vector_store_service.load_document()
    retriever = vector_store_service.get_retriever()
    docs = retriever.invoke("机器学习是什么？")
    for doc in docs:
        print(doc.page_content)
        print("-"*20)