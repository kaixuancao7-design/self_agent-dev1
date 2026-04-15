import sys
sys.path.append('d:/vcodeproject/self_agent-dev1')

print("=" * 60)
print("测试高级Agent功能模块")
print("=" * 60)

# 测试任务拆解模块
print("\n1. 测试任务拆解模块 (TaskDecomposer)")
print("-" * 40)
from agent.modules.task_decomposer import task_decomposer

goal = "帮我制定一个AI行业研究方案"
tasks = task_decomposer.decompose(goal)
print(f"目标: {goal}")
print(f"拆解出 {len(tasks)} 个子任务:")
for i, task in enumerate(tasks, 1):
    print(f"{i}. [{task['id']}] {task['title']} (优先级: {task['priority']})")
    print(f"   描述: {task['description']}")
    if task.get('dependencies'):
        print(f"   依赖: {task['dependencies']}")
    if task.get('tool'):
        print(f"   工具: {task['tool']}")

# 测试路径规划模块
print("\n2. 测试路径规划模块 (PathPlanner)")
print("-" * 40)
from agent.modules.path_planner import path_planner

planned_tasks = path_planner.plan(tasks)
print(f"规划后的执行顺序:")
for i, task in enumerate(planned_tasks, 1):
    print(f"{i}. {task['title']}")

# 测试自我评估模块
print("\n3. 测试自我评估模块 (SelfEvaluator)")
print("-" * 40)
from agent.modules.self_evaluator import self_evaluator

test_task = {"id": "test", "title": "测试任务", "description": "测试评估功能"}
test_result = "这是一个很好的测试结果，内容完整且准确。"
quality, feedback, need_retry = self_evaluator.evaluate(test_task, test_result)
print(f"评估结果:")
print(f"  质量分数: {quality:.2f}")
print(f"  评估意见: {feedback}")
print(f"  需要重试: {need_retry}")

# 测试事实核查模块
print("\n4. 测试事实核查模块 (FactChecker)")
print("-" * 40)
from agent.modules.fact_checker import fact_checker

answer = "AI Agent是2026年最热门的技术趋势之一，市场规模预计达到500亿美元。"
sources = [
    {"page_content": "AI Agent技术在2026年持续升温，成为行业关注焦点。", "metadata": {"source": "行业报告1"}},
    {"page_content": "根据市场研究，AI Agent市场规模预计在2026年达到500亿美元。", "metadata": {"source": "市场分析报告"}}
]
confidence, inconsistencies, suggestion = fact_checker.check(answer, sources)
print(f"核查结果:")
print(f"  置信度: {confidence:.2f}")
print(f"  不一致项: {inconsistencies}")
print(f"  建议: {suggestion}")

# 测试动态调整模块
print("\n5. 测试动态调整模块 (DynamicAdjuster)")
print("-" * 40)
from agent.modules.dynamic_adjuster import dynamic_adjuster

failed_task = {
    "id": "failed_task",
    "title": "检索行业信息",
    "description": "从知识库检索AI行业相关信息",
    "tool": "rag_summarize",
    "params": {"query": "AI行业趋势"}
}
adjustment = dynamic_adjuster.adjust(failed_task, result="", quality_score=0.2)
print(f"调整策略:")
print(f"  动作: {adjustment['action']}")
print(f"  参数: {adjustment.get('params', {})}")
print(f"  消息: {adjustment['message']}")

# 测试完整的 ReactAgent 高级规划
print("\n6. 测试 ReactAgent 高级规划模式")
print("-" * 40)
from agent.react_agent import ReactAgent

agent = ReactAgent()
print(f"高级能力启用状态: {'是' if agent.enable_advanced_features else '否'}")

# 测试任务复杂度分析
test_queries = [
    "AI Agent是什么",
    "帮我制定一个AI行业研究方案",
    "如何优化招聘流程"
]
print("\n任务复杂度分析测试:")
for query in test_queries:
    is_complex = agent._analyze_complexity(query)
    print(f"  '{query}' -> {'复杂任务' if is_complex else '简单任务'}")

print("\n" + "=" * 60)
print("所有模块测试完成!")
print("=" * 60)
