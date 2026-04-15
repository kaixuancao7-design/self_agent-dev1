import sys
sys.path.append('d:/vcodeproject/self_agent-dev1')

print("=" * 60)
print("Testing Web Search Tools")
print("=" * 60)

# 测试导入
print("\n1. Testing Import")
print("-" * 40)
try:
    from agent.tools import web_search, fetch_webpage
    print("[OK] Web search tools imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import: {e}")
    sys.exit(1)

# 测试 web_search
print("\n2. Testing web_search")
print("-" * 40)
print("Searching for: AI Agent latest trends...")
result = web_search.invoke({"query": "AI Agent latest trends 2026", "max_results": 3})
print(f"[OK] Search completed")
print(f"Result length: {len(result)} characters")

import json
try:
    results = json.loads(result)
    print(f"Number of results: {len(results)}")
    for i, item in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Title: {item.get('title')}")
        print(f"  Source: {item.get('source')}")
        print(f"  Snippet: {item.get('snippet')[:50]}...")
except json.JSONDecodeError:
    print(f"Result: {result[:200]}...")

# 测试 fetch_webpage
print("\n3. Testing fetch_webpage")
print("-" * 40)
print("Fetching webpage: https://example.com...")
result = fetch_webpage.invoke({"url": "https://example.com"})
print(f"[OK] Webpage fetched")
print(f"Result: {result[:200]}...")

# 测试 ReactAgent 工具加载
print("\n4. Testing ReactAgent Tool Loading")
print("-" * 40)
from agent.react_agent import ReactAgent
agent = ReactAgent()
tool_names = [t.name for t in agent.tools]
print(f"[OK] ReactAgent created")
print(f"[OK] Total tools: {len(agent.tools)}")
print(f"[OK] Tools: {tool_names}")
print(f"[OK] web_search included: {'web_search' in tool_names}")
print(f"[OK] fetch_webpage included: {'fetch_webpage' in tool_names}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
