import sys
sys.path.append('d:/vcodeproject/self_agent-dev1')
from utils.file_handler import SUPPORTED_FILE_TYPES, get_file_loader, load_file_content

print(f'支持的文件类型: {SUPPORTED_FILE_TYPES}')
print(f'图片类型是否包含: {any(t in SUPPORTED_FILE_TYPES for t in [".png", ".jpg", ".jpeg", ".gif"])}')

test_path = 'test.png'
loader = get_file_loader(test_path)
print(f'图片加载器: {loader}')

test_path2 = 'test.jpg'
loader2 = get_file_loader(test_path2)
print(f'JPG加载器: {loader2}')
