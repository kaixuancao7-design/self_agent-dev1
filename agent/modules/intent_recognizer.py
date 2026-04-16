"""
意图识别模块

实现深度意图理解能力，识别用户的真实需求和隐含意图
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.logger_handler import logger
from model.factory import chat_model


class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self):
        # 预定义意图模板
        self.intent_templates = [
            {
                "intent": "knowledge_query",
                "name": "知识查询",
                "patterns": [
                    "什么是", "什么叫", "解释一下", "定义", "原理", "概念",
                    "如何", "怎么", "方法", "步骤", "教程",
                    "有哪些", "列举", "包含", "分类",
                    "区别", "对比", "差异", "不同"
                ],
                "description": "用户想获取知识或信息"
            },
            {
                "intent": "complaint_analysis",
                "name": "投诉分析",
                "patterns": [
                    "投诉", "问题", "故障", "错误", "失败",
                    "变多了", "增加", "频繁", "严重",
                    "原因", "为什么", "怎么办", "解决"
                ],
                "description": "用户报告问题并寻求解决方案"
            },
            {
                "intent": "task_request",
                "name": "任务请求",
                "patterns": [
                    "帮我", "请帮我", "我需要", "能不能",
                    "制定", "规划", "设计", "编写",
                    "分析", "评估", "总结", "报告"
                ],
                "description": "用户请求执行某个任务"
            },
            {
                "intent": "data_retrieval",
                "name": "数据检索",
                "patterns": [
                    "查询", "查一下", "看看", "检索",
                    "历史", "记录", "统计", "报告"
                ],
                "description": "用户需要检索特定数据或记录"
            },
            {
                "intent": "conversation",
                "name": "闲聊对话",
                "patterns": [
                    "你好", "嗨", "最近怎么样", "聊聊",
                    "在吗", "忙吗", "今天天气", "心情"
                ],
                "description": "用户进行日常对话"
            },
            {
                "intent": "preference_setting",
                "name": "偏好设置",
                "patterns": [
                    "设置", "偏好", "配置", "默认",
                    "习惯", "喜欢", "偏好", "风格"
                ],
                "description": "用户设置或修改偏好"
            }
        ]
        
        # 意图映射到工具
        self.intent_to_tools = {
            "knowledge_query": ["rag_summarize"],
            "complaint_analysis": ["task_decompose", "rag_summarize", "fact_check"],
            "task_request": ["task_decompose", "evaluate_result"],
            "data_retrieval": ["rag_summarize", "get_user_history"],
            "conversation": [],
            "preference_setting": []
        }
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        提取查询中的关键词
        
        :param query: 用户查询
        :return: 关键词列表
        """
        import re
        
        # 移除标点符号
        cleaned = re.sub(r'[，。！？、；：""''（）{}【】<>《》]', ' ', query)
        
        # 分词（简单处理，实际应用中可使用专业分词工具）
        words = cleaned.split()
        
        # 过滤停用词
        stop_words = ["的", "是", "在", "有", "和", "了", "我", "你", "他", "她", "它",
                      "这", "那", "个", "些", "吗", "呢", "啊", "吧", "哦", "嗯",
                      "可以", "会", "能", "要", "不要", "想", "觉得", "认为", "知道"]
        
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords
    
    def match_intent(self, query: str) -> List[Dict[str, Any]]:
        """
        匹配预定义意图
        
        :param query: 用户查询
        :return: 匹配的意图列表（按置信度排序）
        """
        results = []
        keywords = self.extract_keywords(query)
        
        for template in self.intent_templates:
            matched_patterns = []
            for pattern in template["patterns"]:
                if pattern in query:
                    matched_patterns.append(pattern)
            
            if matched_patterns:
                confidence = min(1.0, len(matched_patterns) / len(template["patterns"]))
                
                results.append({
                    "intent": template["intent"],
                    "name": template["name"],
                    "confidence": confidence,
                    "matched_patterns": matched_patterns,
                    "description": template["description"],
                    "suggested_tools": self.intent_to_tools.get(template["intent"], [])
                })
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def analyze_implicit_intent(self, query: str) -> Dict[str, Any]:
        """
        分析隐含意图（使用大模型）
        
        :param query: 用户查询
        :return: 分析结果
        """
        prompt = f"""
        请分析用户查询的真实意图，直接输出JSON格式，不要包含其他内容：
        
        {{
            "intent": "意图名称",
            "confidence": 置信度(0-1),
            "implicit_needs": ["隐含需求1", "隐含需求2"],
            "suggested_actions": ["建议操作1", "建议操作2"],
            "key_points": ["关键点1", "关键点2"]
        }}
        
        用户查询：{query}
        """
        
        try:
            response = chat_model.invoke(prompt)
            content = getattr(response, 'content', str(response)).strip()
            
            # 清理可能的额外内容
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            
            # 解析JSON
            import json
            result = json.loads(content)
            
            # 验证结果结构
            if not isinstance(result, dict):
                raise ValueError("返回结果不是字典")
            if 'intent' not in result:
                result['intent'] = 'unknown'
            if 'confidence' not in result:
                result['confidence'] = 0.7
            
            logger.info(f"[IntentRecognizer] 隐含意图分析完成: {result.get('intent')}")
            return result
        except Exception as e:
            logger.error(f"[IntentRecognizer] 隐含意图分析失败: {e}")
            # 使用规则匹配的结果作为备选
            rule_matches = self.match_intent(query)
            if rule_matches:
                primary = rule_matches[0]
                return {
                    "intent": primary["intent"],
                    "confidence": primary["confidence"],
                    "implicit_needs": [],
                    "suggested_actions": primary.get("suggested_tools", []),
                    "key_points": self.extract_keywords(query)
                }
            return {
                "intent": "unknown",
                "confidence": 0.5,
                "implicit_needs": [],
                "suggested_actions": [],
                "key_points": self.extract_keywords(query)
            }
    
    def recognize(self, query: str) -> Dict[str, Any]:
        """
        综合识别意图
        
        :param query: 用户查询
        :return: 识别结果
        """
        # 1. 规则匹配
        rule_matches = self.match_intent(query)
        
        # 2. 隐含意图分析
        implicit_result = self.analyze_implicit_intent(query)
        
        # 3. 综合结果
        if rule_matches:
            primary_intent = rule_matches[0]
            confidence = (primary_intent["confidence"] + implicit_result["confidence"]) / 2
        else:
            primary_intent = implicit_result
            confidence = implicit_result["confidence"]
        
        result = {
            "query": query,
            "intent": implicit_result.get("intent", primary_intent.get("intent", "unknown")),
            "intent_name": primary_intent.get("name", implicit_result.get("intent", "未知")),
            "confidence": confidence,
            "implicit_needs": implicit_result.get("implicit_needs", []),
            "suggested_actions": implicit_result.get("suggested_actions", []),
            "suggested_tools": primary_intent.get("suggested_tools", []),
            "key_points": implicit_result.get("key_points", []),
            "matched_patterns": primary_intent.get("matched_patterns", []),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[IntentRecognizer] 意图识别完成: {result['intent']} (置信度: {confidence:.2f})")
        
        return result


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()
