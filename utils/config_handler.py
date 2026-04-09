"""
yaml配置文件处理模块
key:value 形式的配置文件解析，提供接口获取配置信息
"""

from utils.path_tool import get_abs_path


def load_rag_config(file_path:str=get_abs_path("config/rag.yml"), encoding:str='utf-8') -> dict:
    """
    加载RAG配置文件，返回一个字典对象。
    :param file_path: 配置文件的路径，默认为config目录下的rag.yml
    :param encoding: 文件编码，默认为utf-8
    :return: 配置字典
    """
    import yaml
    with open(file_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def load_chroma_config(file_path:str=get_abs_path("config/chroma.yml"), encoding:str='utf-8') -> dict:
    """
    加载Chroma配置文件，返回一个字典对象。
    :param file_path: 配置文件的路径，默认为config目录下的chroma.yml
    :param encoding: 文件编码，默认为utf-8
    :return: 配置字典
    """
    import yaml
    with open(file_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
def load_prompts_config(file_path:str=get_abs_path("config/prompts.yml"), encoding:str='utf-8') -> dict:
    """
    加载Prompt配置文件，返回一个字典对象。
    :param file_path: 配置文件的路径，默认为config目录下的prompts.yml
    :param encoding: 文件编码，默认为utf-8
    :return: 配置字典
    """
    import yaml
    with open(file_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
def load_agent_config(file_path:str=get_abs_path("config/agent.yml"), encoding:str='utf-8') -> dict:
    """
    加载Agent配置文件，返回一个字典对象。
    :param file_path: 配置文件的路径，默认为config目录下的agent.yml
    :param encoding: 文件编码，默认为utf-8
    :return: 配置字典
    """
    import yaml
    with open(file_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    

rag_cfg = load_rag_config()
chroma_cfg = load_chroma_config()
prompts_cfg = load_prompts_config()
agent_cfg = load_agent_config()
