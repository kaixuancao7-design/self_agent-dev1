"""
Ragas 评估脚本：用于评估检索与生成效果

使用方法：
python scripts/run_ragas_evaluation.py

功能：
1. 加载测试数据集
2. 运行 RAG 管道评估
3. 生成评估报告
4. 输出 Bad Case 分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ragas_evaluator import ragas_evaluator, evaluate_rag_pipeline, generate_ragas_report
from rag.rag_service import RagSummerizeService
from agent.react_agent import ReactAgent
from utils.logger_handler import logger


def load_test_dataset() -> list:
    """加载测试数据集"""
    test_cases = [
        {
            "query": "什么是机器学习？",
            "expected_topic": "机器学习基础概念"
        },
        {
            "query": "小户型适合什么扫地机器人？",
            "expected_topic": "扫地机器人推荐"
        },
        {
            "query": "AI Agent 有哪些核心能力？",
            "expected_topic": "AI Agent 能力"
        },
        {
            "query": "如何配置 Ollama 本地模型？",
            "expected_topic": "Ollama 配置"
        },
        {
            "query": "RAG 检索流程是怎样的？",
            "expected_topic": "RAG 流程"
        },
        {
            "query": "混合检索策略的优势是什么？",
            "expected_topic": "混合检索"
        },
        {
            "query": "重排策略有哪些类型？",
            "expected_topic": "重排策略"
        },
        {
            "query": "如何添加新的技能？",
            "expected_topic": "技能扩展"
        }
    ]
    return test_cases


def run_evaluation():
    """运行完整评估"""
    logger.info("========== 开始 Ragas 评估 ==========")
    
    # 初始化服务
    try:
        rag_service = RagSummerizeService()
        logger.info("RAG 服务初始化成功")
    except Exception as e:
        logger.error(f"RAG 服务初始化失败: {e}")
        return
    
    # 加载测试数据集
    test_cases = load_test_dataset()
    logger.info(f"加载了 {len(test_cases)} 个测试用例")
    
    # 运行评估
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        logger.info(f"\n--- 测试用例 {i}/{len(test_cases)} ---")
        logger.info(f"查询: {query}")
        
        try:
            # 检索文档
            docs = rag_service.retrieve_docs(query)
            retrieved_docs = []
            for doc in docs:
                retrieved_docs.append({
                    "id": doc.metadata.get('source', str(len(retrieved_docs))),
                    "content": doc.page_content,
                    "score": doc.metadata.get('score', 0.0)
                })
            
            # 生成回答（模拟）
            answer = f"根据检索到的资料，关于 '{query}' 的答案如下：这是一个模拟回答，用于演示评估功能。"
            
            # 评估
            result = evaluate_rag_pipeline(query, answer, retrieved_docs)
            
            # 输出评估结果
            logger.info(f"综合评分: {result['overall_score']:.4f}")
            logger.info(f"是否 Bad Case: {'是' if result['is_bad_case'] else '否'}")
            
            if result["is_bad_case"]:
                logger.warning(f"问题类型: {result['bad_case_analysis']['issue_type']}")
                logger.warning(f"严重程度: {result['bad_case_analysis']['severity']}")
                logger.warning(f"建议: {result['bad_case_analysis']['suggestions']}")
        
        except Exception as e:
            logger.error(f"评估失败: {e}")
    
    # 生成报告
    logger.info("\n========== 生成评估报告 ==========")
    report = generate_ragas_report()
    
    # 打印报告摘要
    print("\n" + "="*50)
    print("RAGAS 评估报告摘要")
    print("="*50)
    print(f"评估时间: {report['report_generated_at']}")
    print(f"评估总数: {report['total_evaluations']}")
    print(f"Bad Case 数量: {report['bad_case_count']}")
    print(f"Bad Case 率: {report['bad_case_rate']:.2%}")
    print(f"平均综合评分: {report['average_overall_score']:.4f}")
    
    print("\n指标汇总:")
    for metric, stats in report['metric_summary'].items():
        print(f"  {metric}: 平均={stats['average']:.4f}, 最小={stats['min']:.4f}, 最大={stats['max']:.4f}")
    
    print("\n问题类型分布:")
    for issue_type, count in report['bad_case_analysis_summary'].get('issue_distribution', {}).items():
        print(f"  {issue_type}: {count}")
    
    print("\n优化建议:")
    for i, recommendation in enumerate(report['optimization_recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    print("\n" + "="*50)
    logger.info("评估完成！")


if __name__ == "__main__":
    run_evaluation()
