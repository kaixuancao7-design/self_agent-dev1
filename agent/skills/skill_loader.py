"""
Skill Loader - 自动发现和加载 Skills
按照 Anthropic Skills 标准设计

新增特性：
- 优化技能自动加载逻辑
- 添加技能失败降级机制（如web_search失败时提示用户）
- 支持技能加载失败后的备选方案
"""
import os
import importlib.util
from typing import Dict, Any, List, Optional, Callable
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from langchain_core.tools import BaseTool


class FallbackSkill:
    """
    降级技能类：当技能加载失败时返回的备选技能
    
    提供友好的错误提示，告知用户技能暂时不可用
    """
    
    def __init__(self, skill_name: str, error: Exception = None):
        self.skill_name = skill_name
        self.error = error
        self.name = f"fallback_{skill_name}"
        
    def __call__(self, **kwargs):
        """执行降级处理"""
        error_msg = str(self.error) if self.error else "未知错误"
        return "抱歉，'{}' 技能暂时不可用。\n\n错误信息：{}\n\n请稍后重试，或尝试使用其他方式完成您的请求。如果问题持续存在，请联系管理员。".format(self.skill_name, error_msg)
    
    def invoke(self, input_data: Dict[str, Any]) -> str:
        """兼容工具调用接口"""
        return self(**input_data)


class SkillLoader:
    """
    Skill 加载器，负责发现和加载 skills 目录中的所有技能
    支持失败降级机制
    """
    
    def __init__(self, enable_fallback: bool = True):
        self.skills_dir = get_abs_path("agent/skills")
        self.skills = {}
        self.skill_metadata = {}
        self.load_errors = {}  # 记录加载失败的技能
        self.enable_fallback = enable_fallback  # 是否启用降级机制
    
    def discover_skills(self) -> List[str]:
        """
        发现 skills 目录中的所有技能
        
        :return: 技能名称列表
        """
        skills = []
        
        if not os.path.exists(self.skills_dir):
            logger.warning(f"[SkillLoader] Skills目录不存在: {self.skills_dir}")
            return skills
        
        for item in os.listdir(self.skills_dir):
            item_path = os.path.join(self.skills_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含 SKILL.md 文件或 script/main.py
                skill_md_path = os.path.join(item_path, "SKILL.md")
                script_path = os.path.join(item_path, "script", "main.py")
                
                if os.path.exists(script_path):
                    skills.append(item)
                    logger.info(f"[SkillLoader] 发现技能: {item}")
                elif os.path.exists(skill_md_path):
                    logger.warning(f"[SkillLoader] 技能 {item} 缺少脚本文件")
        
        return skills
    
    def load_skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        """
        加载技能的元数据（从 SKILL.md）
        
        :param skill_name: 技能名称
        :return: 元数据字典
        """
        md_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
        
        if not os.path.exists(md_path):
            logger.warning(f"[SkillLoader] SKILL.md 不存在: {md_path}")
            return {"name": skill_name}
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "name": skill_name,
                "content": content
            }
            
            lines = content.split('\n')
            in_section = None
            current_section = ""
            
            for line in lines:
                if line.startswith("## "):
                    if in_section:
                        metadata[in_section.lower().replace(' ', '_')] = current_section.strip()
                    in_section = line[3:].strip()
                    current_section = ""
                elif in_section:
                    current_section += line + "\n"
            
            if in_section:
                metadata[in_section.lower().replace(' ', '_')] = current_section.strip()
            
            return metadata
        
        except Exception as e:
            logger.error(f"[SkillLoader] 加载元数据失败 {skill_name}: {e}")
            return {"name": skill_name}
    
    def _get_fallback_skill(self, skill_name: str, error: Exception) -> FallbackSkill:
        """
        获取降级技能
        
        :param skill_name: 技能名称
        :param error: 加载失败的错误
        :return: 降级技能实例
        """
        fallback = FallbackSkill(skill_name, error)
        logger.warning(f"[SkillLoader] 技能 {skill_name} 加载失败，使用降级处理")
        return fallback
    
    def load_skill_function(self, skill_name: str) -> Optional[Callable]:
        """
        加载技能的可执行函数（带失败降级机制）
        
        :param skill_name: 技能名称
        :return: 技能函数或降级处理函数
        """
        script_path = os.path.join(self.skills_dir, skill_name, "script", "main.py")
        
        if not os.path.exists(script_path):
            error = FileNotFoundError(f"脚本文件不存在: {script_path}")
            self.load_errors[skill_name] = str(error)
            if self.enable_fallback:
                return self._get_fallback_skill(skill_name, error)
            logger.error(f"[SkillLoader] 脚本文件不存在: {script_path}")
            return None
        
        try:
            # 使用 importlib 动态导入
            spec = importlib.util.spec_from_file_location(f"skill_{skill_name}", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 查找工具函数（以技能名称命名的函数）
            func_name = skill_name.replace('-', '_')
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                logger.info(f"[SkillLoader] 加载技能函数成功: {skill_name}")
                return func
            else:
                # 查找任意 @tool 装饰的函数
                for name in dir(module):
                    obj = getattr(module, name)
                    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Tool':
                        logger.info(f"[SkillLoader] 加载技能函数成功: {skill_name} -> {name}")
                        return obj
            
            # 查找符合 BaseTool 协议的对象
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, BaseTool):
                    logger.info(f"[SkillLoader] 加载技能工具成功: {skill_name} -> {name}")
                    return obj
            
            error = AttributeError(f"未找到技能函数: {skill_name}")
            self.load_errors[skill_name] = str(error)
            if self.enable_fallback:
                return self._get_fallback_skill(skill_name, error)
            logger.warning(f"[SkillLoader] 未找到技能函数: {skill_name}")
            return None
        
        except Exception as e:
            self.load_errors[skill_name] = str(e)
            logger.error(f"[SkillLoader] 加载技能失败 {skill_name}: {e}", exc_info=True)
            if self.enable_fallback:
                return self._get_fallback_skill(skill_name, e)
            return None
    
    def load_all_skills(self, skip_failed: bool = False) -> Dict[str, Any]:
        """
        加载所有技能
        
        :param skip_failed: 是否跳过加载失败的技能
        :return: 技能字典 {skill_name: {"function": callable, "metadata": dict}}
        """
        skills = {}
        skill_names = self.discover_skills()
        
        for name in skill_names:
            func = self.load_skill_function(name)
            metadata = self.load_skill_metadata(name)
            
            if func:
                skills[name] = {
                    "function": func,
                    "metadata": metadata,
                    "is_fallback": isinstance(func, FallbackSkill)
                }
                self.skill_metadata[name] = metadata
            elif not skip_failed:
                # 记录未加载成功的技能
                skills[name] = {
                    "function": None,
                    "metadata": metadata,
                    "is_fallback": False,
                    "error": self.load_errors.get(name)
                }
        
        success_count = sum(1 for s in skills.values() if s["function"] and not s.get("is_fallback"))
        fallback_count = sum(1 for s in skills.values() if s.get("is_fallback"))
        failed_count = sum(1 for s in skills.values() if s["function"] is None)
        
        logger.info(f"[SkillLoader] 共发现 {len(skill_names)} 个技能")
        logger.info(f"[SkillLoader] 成功加载 {success_count} 个技能")
        logger.info(f"[SkillLoader] 降级处理 {fallback_count} 个技能")
        logger.info(f"[SkillLoader] 加载失败 {failed_count} 个技能")
        
        if self.load_errors:
            logger.warning(f"[SkillLoader] 加载失败的技能: {list(self.load_errors.keys())}")
        
        return skills
    
    def get_tool_list(self, include_fallback: bool = True) -> List:
        """
        获取所有技能的工具列表（用于 Agent）
        
        :param include_fallback: 是否包含降级技能
        :return: 工具函数列表
        """
        skills = self.load_all_skills()
        tools = []
        
        for name, skill in skills.items():
            func = skill.get("function")
            is_fallback = skill.get("is_fallback", False)
            
            if func and (include_fallback or not is_fallback):
                tools.append(func)
        
        return tools
    
    def get_skill_descriptions(self, include_fallback: bool = False) -> str:
        """
        获取所有技能的描述（用于提示词）
        
        :param include_fallback: 是否包含降级技能的描述
        :return: 格式化的技能描述字符串
        """
        descriptions = []
        
        for name, skill in self.skill_metadata.items():
            # 检查是否为降级技能
            if name in self.load_errors and not include_fallback:
                continue
            
            desc = skill.get("capability", "") or skill.get("overview", "") or skill.get("description", "")
            if desc:
                descriptions.append(f"- {name}: {desc}")
            else:
                descriptions.append(f"- {name}: 未知功能")
        
        return "\n".join(descriptions)
    
    def get_loaded_skills_info(self) -> Dict[str, Any]:
        """
        获取已加载技能的详细信息
        
        :return: 包含技能状态的字典
        """
        info = {
            "total_discovered": len(self.discover_skills()),
            "successfully_loaded": [],
            "fallback_skills": [],
            "failed_skills": []
        }
        
        skills = self.load_all_skills()
        
        for name, skill in skills.items():
            if skill.get("is_fallback"):
                info["fallback_skills"].append({
                    "name": name,
                    "error": self.load_errors.get(name)
                })
            elif skill.get("function"):
                info["successfully_loaded"].append(name)
            else:
                info["failed_skills"].append({
                    "name": name,
                    "error": skill.get("error")
                })
        
        return info
    
    def reload_skill(self, skill_name: str) -> bool:
        """
        重新加载指定技能
        
        :param skill_name: 技能名称
        :return: 是否成功加载
        """
        # 清除之前的错误记录
        self.load_errors.pop(skill_name, None)
        
        func = self.load_skill_function(skill_name)
        metadata = self.load_skill_metadata(skill_name)
        
        if func:
            self.skills[skill_name] = {
                "function": func,
                "metadata": metadata,
                "is_fallback": isinstance(func, FallbackSkill)
            }
            self.skill_metadata[skill_name] = metadata
            logger.info(f"[SkillLoader] 重新加载技能成功: {skill_name}")
            return True
        else:
            logger.error(f"[SkillLoader] 重新加载技能失败: {skill_name}")
            return False
    
    def get_fallback_message(self, skill_name: str) -> str:
        """
        获取指定技能的降级提示消息
        
        :param skill_name: 技能名称
        :return: 降级提示消息
        """
        if skill_name in self.load_errors:
            return f"技能 '{skill_name}' 当前不可用，请稍后重试。"
        return f"技能 '{skill_name}' 加载失败。"


# 全局技能加载器
skill_loader = SkillLoader()


def get_skill_loader() -> SkillLoader:
    """
    获取全局技能加载器实例
    
    :return: SkillLoader 实例
    """
    return skill_loader


def load_skills_with_fallback() -> Dict[str, Any]:
    """
    加载所有技能，启用降级机制
    
    :return: 技能字典
    """
    return skill_loader.load_all_skills()


def get_available_tools() -> List:
    """
    获取所有可用工具（包含降级技能）
    
    :return: 工具列表
    """
    return skill_loader.get_tool_list()
