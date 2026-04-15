import sys
sys.path.append('d:/vcodeproject/self_agent-dev1')
from agent.react_agent import ReactAgent

agent = ReactAgent()
response_messages = []
res_stream = agent.execute_stream('测试')

def capture(generator, cache_list):
    for chunk in generator:
        if chunk is None:
            continue
        cache_list.append(chunk)
        if isinstance(chunk, str):
            for char in chunk:
                yield char
        elif isinstance(chunk, dict):
            content = chunk.get('messages', [{}])[-1].get('content', '')
            if content:
                for char in content:
                    yield char
        else:
            content = getattr(chunk, 'content', str(chunk))
            for char in content:
                yield char

for char in capture(res_stream, response_messages):
    pass

print(f'Number of chunks: {len(response_messages)}')
print(f'First 3 chunks: {response_messages[:3]}')
full_response = "".join(response_messages)
print(f'Full response length: {len(full_response)}')
print(f'Full response: {repr(full_response[:100])}')
