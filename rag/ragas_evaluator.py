"""
Ragas 评估模块：用于量化评估检索与生成效果

核心功能：
1. 检索效果评估（Recall、Precision、MRR等）
2. 生成效果评估（Faithfulness、Answer Relevance等）
3. Bad Case 分析与优化建议
4. 分块策略优化指导
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class RagasEvaluator:
    """Ragas 风格的评估器"""
    
    def __init__(self):
        self.evaluation_results = []
        self.bad_cases = []
        self.results_dir = get_abs_path("evaluations")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                          relevant_docs: List[str] = None) -> Dict[str, float]:
        """
        评估检索效果
        
        :param query: 用户查询
        :param retrieved_docs: 检索到的文档列表
        :param relevant_docs: 已知相关文档ID列表（用于计算Recall）
        :return: 评估指标字典
        """
        results = {
            "query": query,
            "retrieved_count": len(retrieved_docs),
            "metrics": {}
        }
        
        if not retrieved_docs:
            results["metrics"]["recall"] = 0.0
            results["metrics"]["precision"] = 0.0
            results["metrics"]["mrr"] = 0.0
            results["metrics"]["hit_rate"] = 0.0
            return results
        
        # 1. Hit Rate（是否检索到至少一条相关文档）
        hit_rate = 1.0 if retrieved_docs else 0.0
        results["metrics"]["hit_rate"] = hit_rate
        
        # 2. Mean Reciprocal Rank (MRR)
        # 简化版本：假设第一个文档最相关
        mrr = 1.0 / 1  # 第一个位置
        if retrieved_docs:
            # 基于相关性分数计算
            scores = [doc.get('score', 0.0) for doc in retrieved_docs]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            mrr = avg_score  # 使用平均分数作为MRR的近似
        results["metrics"]["mrr"] = mrr
        
        # 3. Recall@k（如果提供了相关文档列表）
        if relevant_docs:
            retrieved_ids = [doc.get('id', str(i)) for i, doc in enumerate(retrieved_docs)]
            relevant_found = sum(1 for rid in retrieved_ids if rid in relevant_docs)
            recall = relevant_found / len(relevant_docs) if relevant_docs else 0.0
            results["metrics"]["recall"] = recall
            
            # 4. Precision@k
            precision = relevant_found / len(retrieved_ids) if retrieved_ids else 0.0
            results["metrics"]["precision"] = precision
        else:
            results["metrics"]["recall"] = None
            results["metrics"]["precision"] = None
        
        # 5. 文档多样性（基于内容相似度）
        diversity = self._calculate_diversity(retrieved_docs)
        results["metrics"]["diversity"] = diversity
        
        # 6. 平均相关性分数
        avg_score = sum(doc.get('score', 0.0) for doc in retrieved_docs) / len(retrieved_docs)
        results["metrics"]["avg_relevance_score"] = avg_score
        
        return results
    
    def _calculate_diversity(self, docs: List[Dict[str, Any]]) -> float:
        """计算文档多样性（简化版）"""
        if len(docs) < 2:
            return 1.0 if docs else 0.0
        
        contents = [doc.get('content', '')[:100] for doc in docs]
        # 简单计算：基于内容长度差异
        lengths = [len(c) for c in contents]
        if len(set(lengths)) == 1:
            return 0.3  # 低多样性
        return 0.7  # 中等多样性
    
    def evaluate_generation(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估生成效果
        
        :param query: 用户查询
        :param answer: 生成的答案
        :param retrieved_docs: 检索到的文档列表
        :return: 评估指标字典
        """
        results = {
            "query": query,
            "answer_length": len(answer),
            "metrics": {}
        }
        
        # 1. Faithfulness（事实一致性）
        faithfulness = self._calculate_faithfulness(answer, retrieved_docs)
        results["metrics"]["faithfulness"] = faithfulness
        
        # 2. Answer Relevance（答案相关性）
        relevance = self._calculate_relevance(query, answer)
        results["metrics"]["answer_relevance"] = relevance
        
        # 3. Context Utilization（上下文利用率）
        utilization = self._calculate_context_utilization(answer, retrieved_docs)
        results["metrics"]["context_utilization"] = utilization
        
        # 4. Answer Correctness（答案正确性）- 简化版
        correctness = (faithfulness + relevance) / 2
        results["metrics"]["answer_correctness"] = correctness
        
        # 5. Answer Conciseness（答案简洁性）
        conciseness = self._calculate_conciseness(query, answer)
        results["metrics"]["answer_conciseness"] = conciseness
        
        return results
    
    def _calculate_faithfulness(self, answer: str, docs: List[Dict[str, Any]]) -> float:
        """计算事实一致性"""
        if not docs or not answer:
            return 0.0
        
        # 简单版本：检查答案中的关键信息是否在文档中出现
        doc_text = " ".join(doc.get('content', '') for doc in docs).lower()
        answer_text = answer.lower()
        
        # 提取答案中的关键词
        keywords = [w for w in answer_text.split() if len(w) > 3]
        if not keywords:
            return 0.5  # 中性
        
        # 计算关键词在文档中的覆盖率
        found_keywords = sum(1 for kw in keywords if kw in doc_text)
        return found_keywords / len(keywords)
    
    def _calculate_relevance(self, query: str, answer: str) -> float:
        """计算答案相关性"""
        if not query or not answer:
            return 0.0
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.5
        
        # 计算查询词在答案中的覆盖率
        overlap = query_words.intersection(answer_words)
        return len(overlap) / len(query_words)
    
    def _calculate_context_utilization(self, answer: str, docs: List[Dict[str, Any]]) -> float:
        """计算上下文利用率"""
        if not docs or not answer:
            return 0.0
        
        doc_text = " ".join(doc.get('content', '') for doc in docs)
        answer_length = len(answer)
        context_length = len(doc_text)
        
        # 简单版本：答案长度与上下文长度的比例
        return min(answer_length / context_length, 1.0) if context_length > 0 else 0.0
    
    def _calculate_conciseness(self, query: str, answer: str) -> float:
        """计算答案简洁性"""
        if not query or not answer:
            return 0.5
        
        # 理想情况下，答案长度应该与查询长度成比例
        query_len = len(query)
        answer_len = len(answer)
        
        # 答案长度应该是查询长度的2-10倍比较合理
        ideal_ratio = 5
        actual_ratio = answer_len / query_len if query_len > 0 else 10
        
        # 计算与理想比例的偏差
        deviation = abs(actual_ratio - ideal_ratio) / ideal_ratio
        return max(0.0, 1.0 - deviation)
    
    def analyze_bad_case(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]],
                         evaluation_results: Dict[str, float]) -> Dict[str, Any]:
        """
        分析 Bad Case 并给出优化建议
        
        :param query: 用户查询
        :param answer: 生成的答案
        :param retrieved_docs: 检索到的文档列表
        :param evaluation_results: 评估结果
        :return: Bad Case 分析报告
        """
        analysis = {
            "query": query,
            "issue_type": None,
            "severity": "low",
            "suggestions": [],
            "optimization_actions": []
        }
        
        metrics = evaluation_results.get("metrics", {})
        
        # 分析检索问题
        if metrics.get("hit_rate", 0.0) < 0.5:
            analysis["issue_type"] = "retrieval_failure"
            analysis["severity"] = "high"
            analysis["suggestions"].append("检索未返回任何有效文档")
            analysis["optimization_actions"].extend([
                "检查查询词是否过于模糊或专业术语过多",
                "考虑扩展查询词（同义词、上位词）",
                "检查知识库是否包含相关内容",
                "调整检索参数（如 top_k、相似度阈值）"
            ])
        
        elif metrics.get("avg_relevance_score", 0.0) < 0.5:
            analysis["issue_type"] = "low_relevance"
            analysis["severity"] = "medium"
            analysis["suggestions"].append("检索到的文档相关性较低")
            analysis["optimization_actions"].extend([
                "调整向量检索的相似度阈值",
                "优化 BM25 参数（k1、b）",
                "考虑使用重排策略（Cross-Encoder/LLM）",
                "检查分块策略是否合理"
            ])
        
        # 分析生成问题
        if metrics.get("faithfulness", 0.0) < 0.5:
            analysis["issue_type"] = "hallucination"
            analysis["severity"] = "high"
            analysis["suggestions"].append("答案可能包含幻觉（与参考文档不一致）")
            analysis["optimization_actions"].extend([
                "增强事实核查机制",
                "优化提示词，强调基于文档回答",
                "增加对答案来源的引用要求",
                "考虑使用 RAG 专用提示词模板"
            ])
        
        elif metrics.get("answer_relevance", 0.0) < 0.5:
            analysis["issue_type"] = "irrelevant_answer"
            analysis["severity"] = "medium"
            analysis["suggestions"].append("答案与问题相关性较低")
            analysis["optimization_actions"].extend([
                "优化提示词，明确要求针对问题回答",
                "增加答案相关性检查步骤",
                "考虑在生成前对文档进行摘要"
            ])
        
        elif metrics.get("answer_conciseness", 0.0) < 0.5:
            analysis["issue_type"] = "over_or_under_generation"
            analysis["severity"] = "low"
            analysis["suggestions"].append("答案过于冗长或过于简略")
            analysis["optimization_actions"].extend([
                "在提示词中明确答案长度要求",
                "添加答案长度约束",
                "考虑使用专门的摘要模型"
            ])
        
        # 分析分块问题
        if len(retrieved_docs) > 0:
            avg_chunk_length = sum(len(doc.get('content', '')) for doc in retrieved_docs) / len(retrieved_docs)
            if avg_chunk_length < 100:
                analysis["suggestions"].append("文档分块过小，可能导致上下文不完整")
                analysis["optimization_actions"].append("增大 chunk_size 参数")
            elif avg_chunk_length > 1000:
                analysis["suggestions"].append("文档分块过大，可能包含冗余信息")
                analysis["optimization_actions"].append("减小 chunk_size 参数")
        
        # 如果没有明显问题
        if analysis["issue_type"] is None:
            analysis["issue_type"] = "none"
            analysis["severity"] = "none"
            analysis["suggestions"].append("评估通过，未发现明显问题")
        
        return analysis
    
    def evaluate_qa_pair(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]],
                         relevant_docs: List[str] = None) -> Dict[str, Any]:
        """
        完整评估单个问答对
        
        :param query: 用户查询
        :param answer: 生成的答案
        :param retrieved_docs: 检索到的文档列表
        :param relevant_docs: 已知相关文档ID列表
        :return: 完整评估报告
        """
        # 评估检索效果
        retrieval_eval = self.evaluate_retrieval(query, retrieved_docs, relevant_docs)
        
        # 评估生成效果
        generation_eval = self.evaluate_generation(query, answer, retrieved_docs)
        
        # 合并指标
        combined_metrics = {**retrieval_eval["metrics"], **generation_eval["metrics"]}
        
        # Bad Case 分析
        bad_case_analysis = self.analyze_bad_case(query, answer, retrieved_docs, combined_metrics)
        
        # 综合评分
        overall_score = self._calculate_overall_score(combined_metrics)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "retrieved_docs_count": len(retrieved_docs),
            "metrics": combined_metrics,
            "overall_score": overall_score,
            "bad_case_analysis": bad_case_analysis,
            "is_bad_case": bad_case_analysis["severity"] in ["medium", "high"]
        }
        
        # 保存结果
        self.evaluation_results.append(result)
        
        # 如果是 Bad Case，单独保存
        if result["is_bad_case"]:
            self.bad_cases.append(result)
        
        return result
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        weights = {
            "hit_rate": 0.15,
            "mrr": 0.15,
            "recall": 0.10,
            "precision": 0.10,
            "diversity": 0.05,
            "avg_relevance_score": 0.10,
            "faithfulness": 0.20,
            "answer_relevance": 0.10,
            "context_utilization": 0.03,
            "answer_correctness": 0.02
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            value = metrics.get(metric)
            if value is not None:
                score += value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        生成评估报告
        
        :param save_to_file: 是否保存到文件
        :return: 评估报告
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        # 计算汇总统计
        all_metrics = [r["metrics"] for r in self.evaluation_results]
        bad_case_count = sum(1 for r in self.evaluation_results if r["is_bad_case"])
        
        report = {
            "report_generated_at": datetime.now().isoformat(),
            "total_evaluations": len(self.evaluation_results),
            "bad_case_count": bad_case_count,
            "bad_case_rate": bad_case_count / len(self.evaluation_results),
            "average_overall_score": sum(r["overall_score"] for r in self.evaluation_results) / len(self.evaluation_results),
            "metric_summary": {},
            "bad_case_analysis_summary": {},
            "optimization_recommendations": []
        }
        
        # 计算每个指标的平均值
        metric_names = ["hit_rate", "mrr", "recall", "precision", "diversity", 
                       "avg_relevance_score", "faithfulness", "answer_relevance",
                       "context_utilization", "answer_correctness", "answer_conciseness"]
        
        for metric in metric_names:
            values = [m.get(metric) for m in all_metrics if m.get(metric) is not None]
            if values:
                report["metric_summary"][metric] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # 汇总 Bad Case 类型分布
        issue_types = {}
        for r in self.evaluation_results:
            issue_type = r["bad_case_analysis"]["issue_type"]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        report["bad_case_analysis_summary"]["issue_distribution"] = issue_types
        
        # 生成优化建议
        report["optimization_recommendations"] = self._generate_optimization_recommendations()
        
        # 保存报告
        if save_to_file:
            filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"[RagasEvaluator] 评估报告已保存: {filepath}")
        
        return report
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于评估结果生成建议
        if self.evaluation_results:
            avg_faithfulness = sum(r["metrics"].get("faithfulness", 0.5) for r in self.evaluation_results) / len(self.evaluation_results)
            avg_relevance = sum(r["metrics"].get("answer_relevance", 0.5) for r in self.evaluation_results) / len(self.evaluation_results)
            avg_hit_rate = sum(r["metrics"].get("hit_rate", 0.5) for r in self.evaluation_results) / len(self.evaluation_results)
            
            if avg_faithfulness < 0.7:
                recommendations.append("【高优先级】优化提示词，增强事实一致性约束，考虑添加事实核查步骤")
            
            if avg_relevance < 0.7:
                recommendations.append("【高优先级】优化提示词，明确要求答案与问题强相关")
            
            if avg_hit_rate < 0.8:
                recommendations.append("【中优先级】优化检索策略，考虑调整 top_k、使用重排、扩展查询词")
            
            # 通用建议
            recommendations.append("【持续优化】定期收集 Bad Case，分析模式并针对性优化")
            recommendations.append("【持续优化】根据 Bad Case 分析结果调整分块策略")
            recommendations.append("【可选】考虑引入专用的重排模型（如 Cross-Encoder）提升检索效果")
        
        return recommendations
    
    def load_evaluation_results(self, filepath: str) -> bool:
        """加载评估结果文件"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.evaluation_results.extend(data)
                elif isinstance(data, dict) and "evaluation_results" in data:
                    self.evaluation_results.extend(data["evaluation_results"])
            logger.info(f"[RagasEvaluator] 已加载 {len(data)} 条评估结果")
            return True
        except Exception as e:
            logger.error(f"[RagasEvaluator] 加载评估结果失败: {e}")
            return False
    
    def get_bad_cases(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取 Bad Case 列表"""
        if severity is None:
            return self.bad_cases
        return [case for case in self.bad_cases if case["bad_case_analysis"]["severity"] == severity]


# 全局评估器实例
ragas_evaluator = RagasEvaluator()


def evaluate_rag_pipeline(query: str, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    便捷函数：评估 RAG 管道效果
    
    :param query: 用户查询
    :param answer: 生成的答案
    :param retrieved_docs: 检索到的文档列表
    :return: 评估结果
    """
    return ragas_evaluator.evaluate_qa_pair(query, answer, retrieved_docs)


def generate_ragas_report() -> Dict[str, Any]:
    """便捷函数：生成评估报告"""
    return ragas_evaluator.generate_report()
