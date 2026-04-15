import sys
sys.path.append('d:/vcodeproject/self_agent-dev1')

print("=" * 60)
print("Testing Tools Integration")
print("=" * 60)

# 测试1: 导入所有工具
print("\n1. Testing Tool Imports")
print("-" * 40)
try:
    from agent.tools import (
        rag_summarize,
        get_weather,
        get_user_location,
        get_user_id,
        get_current_month,
        fill_context_report,
        get_user_history,
        task_decompose,
        evaluate_result,
        fact_check
    )
    print("[OK] All tools imported successfully")
    print(f"Tools: rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, fill_context_report, get_user_history, task_decompose, evaluate_result, fact_check")
except Exception as e:
    print(f"[ERROR] Failed to import tools: {e}")

# 测试2: 测试工具描述
print("\n2. Testing Tool Descriptions")
print("-" * 40)
tools_info = [
    ("rag_summarize", rag_summarize),
    ("get_weather", get_weather),
    ("get_user_location", get_user_location),
    ("get_user_id", get_user_id),
    ("get_current_month", get_current_month),
    ("fill_context_report", fill_context_report),
    ("get_user_history", get_user_history),
    ("task_decompose", task_decompose),
    ("evaluate_result", evaluate_result),
    ("fact_check", fact_check),
]

for name, tool_func in tools_info:
    description = getattr(tool_func, 'description', 'No description')
    print(f"{name}: {description[:50]}...")

# 测试3: 测试通用工具
print("\n3. Testing Utility Tools")
print("-" * 40)

print("Testing get_weather...")
weather_result = get_weather.invoke({"city": "Beijing"})
print(f"  Result: {weather_result}")

print("Testing get_user_location...")
location_result = get_user_location.invoke({})
print(f"  Result: {location_result}")

print("Testing get_user_id...")
user_id_result = get_user_id.invoke({})
print(f"  Result: {user_id_result}")

print("Testing get_current_month...")
month_result = get_current_month.invoke({})
print(f"  Result: {month_result}")

# 测试4: 测试高级能力工具
print("\n4. Testing Advanced Tools")
print("-" * 40)

print("Testing task_decompose...")
decompose_result = task_decompose.invoke({"goal": "Help me create a study plan"})
print(f"  Result length: {len(decompose_result)} chars")

print("Testing evaluate_result...")
eval_result = evaluate_result.invoke({
    "task_description": "Test task",
    "result": "This is a good test result"
})
print(f"  Result: {eval_result}")

print("Testing fact_check...")
import json
sources = json.dumps([
    {"page_content": "AI Agent is a hot technology in 2026", "metadata": {"source": "test"}}
])
check_result = fact_check.invoke({
    "answer": "AI Agent is the hottest technology trend in 2026",
    "sources": sources
})
print(f"  Result: {check_result}")

# 测试5: 测试 ReactAgent 工具加载
print("\n5. Testing ReactAgent Tool Loading")
print("-" * 40)
from agent.react_agent import ReactAgent
agent = ReactAgent()
print(f"[OK] ReactAgent created successfully")
print(f"[OK] Number of tools: {len(agent.tools)}")
print(f"[OK] Advanced features enabled: {'Yes' if agent.enable_advanced_features else 'No'}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
