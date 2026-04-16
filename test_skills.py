"""测试技能加载"""
from agent.skills.skill_loader import skill_loader

# 加载所有技能
tools = skill_loader.get_tool_list()

print(f"加载的工具数量: {len(tools)}")
print("工具名称列表:")
for t in tools:
    name = getattr(t, "name", "unknown")
    print(f"  - {name}")

# 测试 content_generator
print("\n测试 content_generator:")
try:
    from agent.skills.content_generator.script.main import content_generator
    result = content_generator.invoke({
        "task_type": "summary",
        "content": "AI Agent技术是2026年最热门的技术趋势之一，具有自主决策能力、多模态整合和垂直领域落地加速等特点。"
    })
    print(f"  成功！结果长度: {len(result)}")
except Exception as e:
    print(f"  失败: {e}")

# 测试 document_processor
print("\n测试 document_processor:")
try:
    from agent.skills.document_processor.script.main import document_processor
    result = document_processor.invoke({
        "action": "analyze",
        "content": "测试文档内容\n\n## 标题\n这是一段测试文本。"
    })
    print(f"  成功！结果长度: {len(result)}")
except Exception as e:
    print(f"  失败: {e}")

# 测试 knowledge_governance
print("\n测试 knowledge_governance:")
try:
    from agent.skills.knowledge_governance.script.main import knowledge_governance
    result = knowledge_governance.invoke({
        "action": "generate_report"
    })
    print(f"  成功！结果长度: {len(result)}")
except Exception as e:
    print(f"  失败: {e}")

print("\n所有测试完成！")
