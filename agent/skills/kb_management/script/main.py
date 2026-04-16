"""
Knowledge Base Management Skill - 知识库管理操作
"""
import json
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="知识库管理操作：获取统计、列表、创建、删除、切换数据库等")
def kb_management(action: str, db_name: str = None, md5: str = None, 
                  file_path: str = None, file_name: str = None) -> str:
    """
    知识库管理操作
    
    :param action: 操作类型 (get_stats, list_dbs, create_db, delete_db, switch_db, list_files, remove_file, reparse_file)
    :param db_name: 数据库名称
    :param md5: 文件MD5值
    :param file_path: 文件路径
    :param file_name: 文件名
    :return: JSON格式的操作结果
    """
    logger.info(f"[kb_management] 执行操作: {action}")
    
    try:
        from rag.vector_store import VectorStoreService
        
        store = VectorStoreService()
        
        if action == "get_stats":
            result = store.get_collection_stats()
        elif action == "list_dbs":
            dbs = VectorStoreService.list_databases()
            result = {"success": True, "message": "获取成功", "data": {"databases": dbs}}
        elif action == "create_db":
            result = store.create_database(db_name)
        elif action == "delete_db":
            store = VectorStoreService(db_name=db_name)
            result = store.delete_database()
        elif action == "switch_db":
            result = store.switch_database(db_name)
        elif action == "list_files":
            files = store.get_uploaded_files()
            result = {"success": True, "message": "获取成功", "data": {"files": files}}
        elif action == "remove_file":
            result = store.remove_file(md5)
        elif action == "reparse_file":
            result = store.reparse_file(file_path, file_name)
        else:
            result = {"success": False, "message": f"未知操作类型: {action}"}
        
        logger.info(f"[kb_management] 操作完成: {result.get('message', '未知')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"[kb_management] 操作失败: {e}", exc_info=True)
        return json.dumps({"success": False, "message": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 测试
    result = kb_management.invoke({"action": "get_stats"})
    print(result)
