"""
提示词相关函数
"""
from utils.config_handler import prompts_cfg
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

def load_system_prompt() -> str:
    try:
        system_prompt_path = get_abs_path(prompts_cfg["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[加载主提示词]配置文件中未找到主提示词路径：{e}")
        raise e
    try:
        return open(system_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[加载主提示词]加载提示词文件失败：{str(e)}")
        raise e
    
def load_rag_prompt() -> str:
    try:
        rag_prompt_path = get_abs_path(prompts_cfg["rag_summary_prompt_path"])
    except KeyError as e:
        logger.error(f"[加载RAG提示词]配置文件中未找到RAG提示词路径：{e}")
        raise e
    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[加载RAG提示词]加载提示词文件失败：{str(e)}")
        raise e
    
def load_report_prompt() -> str:
    try:
        report_prompt_path = get_abs_path(prompts_cfg["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[加载报告提示词]配置文件中未找到报告提示词路径：{e}")
        raise e
    try:
        return open(report_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[加载报告提示词]加载提示词文件失败：{str(e)}")
        raise e
    
if __name__ == "__main__":
    print(load_system_prompt())
    print(load_rag_prompt())
    print(load_report_prompt())