"""
Celery 任务队列 - 处理长任务

支持的任务类型：
- 文件上传和解析
- RAGAS 评估
- 大文件处理
"""

import os
import time
from celery import Celery
from celery.utils.log import get_task_logger
from rag.ragas_evaluator import evaluate_rag_pipeline

# 配置 Celery
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1小时超时
    task_soft_time_limit=300,  # 5分钟软超时
)

logger = get_task_logger(__name__)


@app.task(bind=True)
def process_large_file(self, file_path: str, file_name: str, vector_store_config: dict = None):
    """
    处理大文件上传
    
    :param self: 任务实例
    :param file_path: 文件路径
    :param file_name: 文件名
    :param vector_store_config: 向量存储配置
    :return: 处理结果
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': '开始处理文件'})
        
        # 导入向量存储
        from rag.vector_store import ChromaVectorStore
        
        self.update_state(state='PROGRESS', meta={'progress': 20, 'message': '加载向量存储'})
        
        # 创建向量存储实例
        vector_store = ChromaVectorStore()
        
        self.update_state(state='PROGRESS', meta={'progress': 30, 'message': '解析文件内容'})
        
        # 处理文件
        result = vector_store.upload_file(file_path, file_name)
        
        self.update_state(state='PROGRESS', meta={'progress': 80, 'message': '文件处理完成'})
        
        if result.get('success', False):
            return {
                'status': 'success',
                'progress': 100,
                'message': f'文件 "{file_name}" 处理成功',
                'data': result.get('data', {})
            }
        else:
            return {
                'status': 'failed',
                'progress': 100,
                'message': result.get('message', '处理失败')
            }
            
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        return {
            'status': 'failed',
            'progress': 100,
            'message': str(e)
        }


@app.task(bind=True)
def evaluate_ragas(self, query: str, answer: str, contexts: list, model_config: dict = None):
    """
    RAGAS 评估任务
    
    :param self: 任务实例
    :param query: 用户查询
    :param answer: 生成的答案
    :param contexts: 检索到的上下文
    :param model_config: 模型配置
    :return: 评估结果
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': '开始评估'})
        
        # 执行评估
        self.update_state(state='PROGRESS', meta={'progress': 30, 'message': '运行 RAGAS 评估'})
        
        result = evaluate_rag_pipeline(query, answer, contexts)
        
        self.update_state(state='PROGRESS', meta={'progress': 90, 'message': '评估完成'})
        
        return {
            'status': 'success',
            'progress': 100,
            'message': '评估完成',
            'data': result
        }
        
    except Exception as e:
        logger.error(f"RAGAS 评估失败: {e}")
        return {
            'status': 'failed',
            'progress': 100,
            'message': str(e)
        }


@app.task(bind=True)
def batch_process_files(self, files: list, vector_store_config: dict = None):
    """
    批量处理文件
    
    :param self: 任务实例
    :param files: 文件列表 [{'path': 'xxx', 'name': 'xxx'}, ...]
    :param vector_store_config: 向量存储配置
    :return: 批量处理结果
    """
    try:
        total_files = len(files)
        results = []
        
        for i, file_info in enumerate(files):
            progress = int((i / total_files) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'message': f'处理文件 {i+1}/{total_files}: {file_info["name"]}'
                }
            )
            
            # 处理单个文件
            result = process_large_file.apply(
                args=(file_info['path'], file_info['name'], vector_store_config)
            ).get()
            
            results.append(result)
            
            # 模拟处理时间
            time.sleep(0.5)
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        
        return {
            'status': 'success' if success_count == total_files else 'partial',
            'progress': 100,
            'message': f'批量处理完成: {success_count}/{total_files} 成功',
            'results': results
        }
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        return {
            'status': 'failed',
            'progress': 100,
            'message': str(e)
        }


@app.task(bind=True)
def document_chunking(self, document_text: str, chunk_size: int = 512, chunk_overlap: int = 128):
    """
    文档分块任务
    
    :param self: 任务实例
    :param document_text: 文档文本
    :param chunk_size: 块大小
    :param chunk_overlap: 重叠大小
    :return: 分块结果
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 10, 'message': '开始分块'})
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(document_text)
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'message': '分块完成'})
        
        return {
            'status': 'success',
            'progress': 100,
            'message': f'分块完成，共 {len(chunks)} 块',
            'data': {
                'chunks': chunks,
                'chunk_count': len(chunks),
                'avg_chunk_length': sum(len(c) for c in chunks) // len(chunks) if chunks else 0
            }
        }
        
    except Exception as e:
        logger.error(f"文档分块失败: {e}")
        return {
            'status': 'failed',
            'progress': 100,
            'message': str(e)
        }


@app.task
def cleanup_temp_files(file_paths: list):
    """
    清理临时文件
    
    :param file_paths: 文件路径列表
    """
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"已删除临时文件: {file_path}")
        return {'status': 'success', 'message': '清理完成'}
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")
        return {'status': 'failed', 'message': str(e)}


# 任务状态查询辅助函数
def get_task_status(task_id: str) -> dict:
    """
    获取任务状态
    
    :param task_id: 任务ID
    :return: 任务状态信息
    """
    result = app.AsyncResult(task_id)
    
    if result.state == 'PENDING':
        return {
            'task_id': task_id,
            'status': 'pending',
            'progress': 0,
            'message': '任务等待中'
        }
    elif result.state == 'PROGRESS':
        return {
            'task_id': task_id,
            'status': 'running',
            'progress': result.info.get('progress', 0),
            'message': result.info.get('message', '处理中')
        }
    elif result.state == 'SUCCESS':
        return {
            'task_id': task_id,
            'status': 'completed',
            'progress': 100,
            'message': result.info.get('message', '完成'),
            'data': result.info.get('data')
        }
    elif result.state == 'FAILURE':
        return {
            'task_id': task_id,
            'status': 'failed',
            'progress': 100,
            'message': str(result.info),
            'error': str(result.info)
        }
    else:
        return {
            'task_id': task_id,
            'status': result.state.lower(),
            'progress': 0,
            'message': str(result.info)
        }