"""
通用工具
包含天气、定位、用户ID、时间等通用功能
"""
from langchain_core.tools import tool
from utils.config_handler import agent_cfg
from utils.logger_handler import logger
import random
from datetime import datetime


@tool(description="获取指定城市的天气信息")
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    
    :param city: 城市名称
    :return: 天气信息字符串
    """
    logger.info(f"[get_weather] 获取城市天气: {city}")
    # 后续可接入真实天气API
    return f"{city}的天气是晴天，温度25度，湿度60%，风速10公里每小时。"


@tool(description="获取用户定位信息")
def get_user_location() -> str:
    """
    获取用户定位信息
    
    :return: 用户位置字符串
    """
    # 后续可接入真实定位API
    locations = agent_cfg.get("user_locations", [
        "北京市朝阳区", 
        "上海市浦东新区", 
        "广州市天河区", 
        "深圳市南山区"
    ])
    location = random.choice(locations)
    logger.info(f"[get_user_location] 获取用户位置: {location}")
    return f"用户位于{location}。"


@tool(description="获取用户ID")
def get_user_id() -> str:
    """
    获取用户ID
    
    :return: 用户ID字符串
    """
    # 后续可接入真实用户ID生成
    # 这里简单模拟一个随机用户ID生成
    user_ids = agent_cfg.get("user_ids", [
        "USER_001", 
        "USER_002", 
        "USER_003", 
        "USER_004", 
        "USER_005"
    ])
    user_id = random.choice(user_ids)
    logger.info(f"[get_user_id] 获取用户ID: {user_id}")
    return user_id


@tool(description="获取当前月份")
def get_current_month() -> str:
    """
    获取当前月份（返回真实时间）
    
    :return: 当前月份字符串
    """
    now = datetime.now()
    month_str = f"{now.year}年{now.month}月"
    logger.info(f"[get_current_month] 获取当前月份: {month_str}")
    return f"当前月份是{month_str}。"


# ==================== 知识库管理工具 ====================

@tool(description="获取当前知识库的统计信息")
def get_knowledge_base_stats() -> str:
    """
    获取当前知识库的统计信息，包括总chunk数、文件数等
    
    :return: 统计信息字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService()
        stats = store.get_collection_stats()
        if stats.get("success"):
            result = f"知识库统计信息：\n" \
                     f"- 当前数据库：{stats.get('db_name', '未知')}\n" \
                     f"- 总Chunk数：{stats.get('total_chunks', 0)}\n" \
                     f"- 已上传文件数：{stats.get('total_files', 0)}"
        else:
            result = f"获取统计信息失败：{stats.get('message', '未知错误')}"
        logger.info(f"[get_knowledge_base_stats] 获取统计信息: {result}")
        return result
    except Exception as e:
        logger.error(f"[get_knowledge_base_stats] 获取统计信息失败: {e}", exc_info=True)
        return f"获取知识库统计信息失败：{str(e)}"


@tool(description="获取所有可用的知识库数据库列表")
def list_databases() -> str:
    """
    获取所有可用的知识库数据库列表
    
    :return: 数据库列表字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        dbs = VectorStoreService.list_databases()
        if dbs:
            result = "可用的知识库数据库列表：\n"
            for i, db in enumerate(dbs, 1):
                result += f"{i}. {db}\n"
        else:
            result = "暂无可用的知识库数据库。"
        logger.info(f"[list_databases] 获取数据库列表: {result}")
        return result
    except Exception as e:
        logger.error(f"[list_databases] 获取数据库列表失败: {e}", exc_info=True)
        return f"获取数据库列表失败：{str(e)}"


@tool(description="创建新的知识库数据库")
def create_database(db_name: str) -> str:
    """
    创建新的知识库数据库
    
    :param db_name: 数据库名称
    :return: 创建结果字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService()
        result = store.create_database(db_name)
        if result.get("success"):
            return f"数据库创建成功：{result.get('message')}"
        else:
            return f"数据库创建失败：{result.get('message')}"
    except Exception as e:
        logger.error(f"[create_database] 创建数据库失败: {e}", exc_info=True)
        return f"创建数据库失败：{str(e)}"


@tool(description="删除指定的知识库数据库")
def delete_database(db_name: str) -> str:
    """
    删除指定的知识库数据库（需谨慎操作）
    
    :param db_name: 数据库名称
    :return: 删除结果字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService(db_name=db_name)
        result = store.delete_database()
        if result.get("success"):
            return f"数据库删除成功：{result.get('message')}"
        else:
            return f"数据库删除失败：{result.get('message')}"
    except Exception as e:
        logger.error(f"[delete_database] 删除数据库失败: {e}", exc_info=True)
        return f"删除数据库失败：{str(e)}"


@tool(description="切换到指定的知识库数据库")
def switch_database(db_name: str) -> str:
    """
    切换到指定的知识库数据库
    
    :param db_name: 数据库名称
    :return: 切换结果字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService()
        result = store.switch_database(db_name)
        if result.get("success"):
            return f"数据库切换成功：{result.get('message')}"
        else:
            return f"数据库切换失败：{result.get('message')}"
    except Exception as e:
        logger.error(f"[switch_database] 切换数据库失败: {e}", exc_info=True)
        return f"切换数据库失败：{str(e)}"


@tool(description="获取当前知识库中已上传的文件列表")
def list_uploaded_files() -> str:
    """
    获取当前知识库中已上传的文件列表
    
    :return: 文件列表字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService()
        files = store.get_uploaded_files()
        if files:
            result = "已上传的文件列表：\n"
            for i, file_info in enumerate(files, 1):
                result += f"{i}. {file_info.get('file_name', '未知文件名')} " \
                          f"(Chunks: {file_info.get('chunks', 0)}, " \
                          f"上传时间: {file_info.get('uploaded_at', '未知')})\n"
        else:
            result = "当前知识库暂无已上传的文件。"
        logger.info(f"[list_uploaded_files] 获取文件列表: {result}")
        return result
    except Exception as e:
        logger.error(f"[list_uploaded_files] 获取文件列表失败: {e}", exc_info=True)
        return f"获取文件列表失败：{str(e)}"


@tool(description="从知识库中删除指定文件（通过MD5值）")
def remove_file_from_knowledge_base(md5: str) -> str:
    """
    从知识库中删除指定MD5值的文件
    
    :param md5: 文件的MD5值
    :return: 删除结果字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService()
        result = store.remove_file(md5)
        if result.get("success"):
            return f"文件删除成功：{result.get('message')}"
        else:
            return f"文件删除失败：{result.get('message')}"
    except Exception as e:
        logger.error(f"[remove_file_from_knowledge_base] 删除文件失败: {e}", exc_info=True)
        return f"删除文件失败：{str(e)}"


@tool(description="重新解析知识库中的指定文件")
def reparse_file_in_knowledge_base(file_path: str, file_name: str) -> str:
    """
    重新解析指定文件（先删除旧的，再重新上传）
    
    :param file_path: 文件路径
    :param file_name: 文件名
    :return: 重新解析结果字符串
    """
    from rag.vector_store import VectorStoreService
    try:
        store = VectorStoreService()
        result = store.reparse_file(file_path, file_name)
        if result.get("success"):
            return f"文件重新解析成功：{result.get('message')}"
        else:
            return f"文件重新解析失败：{result.get('message')}"
    except Exception as e:
        logger.error(f"[reparse_file_in_knowledge_base] 重新解析文件失败: {e}", exc_info=True)
        return f"重新解析文件失败：{str(e)}"
