"""
Knowledge Governance Skill - 持续知识治理
能够主动维护知识库的健康状态，包括质量报告、批量更新等功能
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.tools import tool
from utils.logger_handler import logger


@tool(description="知识治理：主动维护知识库健康状态，包括质量报告、统计分析等")
def knowledge_governance(action: str, query: str = None, update_content: str = None) -> str:
    """
    知识治理操作
    
    :param action: 操作类型 (generate_report: 生成质量报告, analyze_stats: 分析统计信息)
    :param query: 查询词（保留参数，用于扩展）
    :param update_content: 更新内容（保留参数，用于扩展）
    :return: JSON格式的操作结果
    """
    logger.info(f"[knowledge_governance] 执行操作: {action}, 查询: {query}")
    
    try:
        from rag.vector_store import VectorStoreService
        
        store = VectorStoreService()
        
        if action == "generate_report":
            result = generate_quality_report(store)
        
        elif action == "analyze_stats":
            result = analyze_statistics(store)
        
        else:
            return json.dumps({
                "success": False, 
                "message": f"未知操作类型: {action}，支持的类型: generate_report, analyze_stats"
            }, ensure_ascii=False)
        
        logger.info(f"[knowledge_governance] 操作完成: {result.get('message', '未知')}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        logger.error(f"[knowledge_governance] 操作失败: {e}", exc_info=True)
        return json.dumps({"success": False, "message": str(e)}, ensure_ascii=False)


def generate_quality_report(store) -> Dict[str, Any]:
    """
    生成知识库质量报告
    """
    try:
        stats = store.get_collection_stats()
        uploaded_files = store.get_uploaded_files()
        
        # 计算健康分数
        total_chunks = stats.get("total_chunks", 0)
        total_files = stats.get("total_files", 0)
        
        health_score = calculate_health_score(total_chunks, total_files)
        recommendations = generate_recommendations(total_chunks, total_files)
        
        report = {
            "success": True,
            "message": "质量报告生成完成",
            "generated_at": datetime.now().isoformat(),
            "basic_stats": stats,
            "quality_metrics": {
                "total_chunks": total_chunks,
                "total_files": total_files,
                "health_score": health_score
            },
            "recommendations": recommendations
        }
        
        return report
    
    except Exception as e:
        logger.error(f"[knowledge_governance] 生成报告失败: {e}")
        return {"success": False, "message": str(e)}


def analyze_statistics(store) -> Dict[str, Any]:
    """
    分析知识库统计信息
    """
    try:
        stats = store.get_collection_stats()
        uploaded_files = store.get_uploaded_files()
        
        # 分析文件类型分布
        file_type_distribution = {}
        for file_info in uploaded_files:
            if isinstance(file_info, dict):
                file_name = file_info.get("name", "")
            else:
                file_name = str(file_info)
            
            ext = os.path.splitext(file_name)[1].lower() if file_name else ".unknown"
            file_type_distribution[ext] = file_type_distribution.get(ext, 0) + 1
        
        analysis = {
            "success": True,
            "message": "统计分析完成",
            "analyzed_at": datetime.now().isoformat(),
            "stats": stats,
            "file_type_distribution": file_type_distribution,
            "insights": generate_insights(stats, file_type_distribution)
        }
        
        return analysis
    
    except Exception as e:
        logger.error(f"[knowledge_governance] 分析统计失败: {e}")
        return {"success": False, "message": str(e)}


def calculate_health_score(total_chunks: int, total_files: int) -> float:
    """
    计算知识库健康分数
    """
    if total_chunks == 0 and total_files == 0:
        return 0.0
    
    # 基础分
    score = 50
    
    # 文档数量加分
    chunk_bonus = min(total_chunks * 0.5, 30)
    file_bonus = min(total_files * 2, 20)
    
    score = score + chunk_bonus + file_bonus
    
    return max(0, min(100, score))


def generate_recommendations(total_chunks: int, total_files: int) -> List[str]:
    """
    生成改进建议
    """
    recommendations = []
    
    if total_chunks == 0:
        recommendations.append("知识库为空，建议上传文档")
    
    if total_files == 0:
        recommendations.append("暂无上传文件，建议上传知识文档")
    
    if total_files < 5:
        recommendations.append("文件数量较少，建议增加更多知识文档")
    
    if total_chunks < 50:
        recommendations.append("Chunk数量较少，建议上传更多内容丰富的文档")
    
    if not recommendations:
        recommendations.append("知识库状态良好，继续保持")
    
    return recommendations


def generate_insights(stats: Dict[str, Any], file_types: Dict[str, int]) -> List[str]:
    """
    生成洞察分析
    """
    insights = []
    
    total_files = sum(file_types.values())
    
    if total_files > 0:
        # 最常见的文件类型
        most_common = max(file_types, key=file_types.get)
        insights.append(f"最常见的文件类型: {most_common} ({file_types[most_common]}个)")
        
        # 文件类型多样性
        if len(file_types) >= 5:
            insights.append("知识库包含多种类型的文档，内容多样性良好")
        elif len(file_types) == 1:
            insights.append("知识库仅包含单一类型文档，建议增加文档类型多样性")
    
    chunks_per_file = stats.get("total_chunks", 0) / max(total_files, 1)
    insights.append(f"平均每个文件生成 {round(chunks_per_file, 1)} 个Chunk")
    
    return insights


if __name__ == "__main__":
    # 测试
    result = knowledge_governance.invoke({"action": "generate_report"})
    print(result)
