#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修复：循环依赖风险
"""
import sys
import os

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import_no_cycle():
    """测试循环依赖修复"""
    print("=== 测试循环依赖修复 ===")
    try:
        from utils.prompt_loader import load_system_prompt, load_rag_prompt
        print("✓ prompt_loader 导入成功")
        
        from utils.prompt_manager import DynamicPromptGenerator, FeedbackCollector
        print("✓ prompt_manager 导入成功")
        
        prompt = load_system_prompt("knowledge_query")
        print(f"✓ 动态提示词生成长度: {len(prompt)} 字符")
        
        generator = DynamicPromptGenerator()
        context = {"name": "测试", "place": "AI"}
        result = generator._replace_template_variables("Hello {{name}}!", context)
        print(f"✓ 模板替换正常: {result}")
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

def test_feedback_validation():
    """测试反馈验证"""
    print("\n=== 测试反馈记录数据验证 ===")
    try:
        from utils.prompt_manager import FeedbackCollector
        collector = FeedbackCollector()
        
        # 测试评分超出范围
        try:
            collector.record_feedback(prompt_version="v1.0", query="test", response="test", rating=0)
            print("✗ 未拒绝无效评分(0)")
            return False
        except ValueError as e:
            print(f"✓ 正确拒绝无效评分: {e}")
            
        # 测试正常记录
        collector.record_feedback(prompt_version="v1.0", query="test", response="test", rating=5)
        print("✓ 正常反馈记录成功")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    return True

if __name__ == "__main__":
    results = [test_import_no_cycle(), test_feedback_validation()]
    if all(results):
        print("\n✓ 所有修复验证通过！")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败")
        sys.exit(1)