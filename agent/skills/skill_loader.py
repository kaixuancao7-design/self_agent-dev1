"""
Skill Loader - 自动发现和加载 Skills
按照 Anthropic Skills 标准设计
"""
import os
import importlib.util
from typing import Dict, Any, List
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class SkillLoader:
    """
    Skill 加载器，负责发现和加载 skills 目录中的所有技能
    """
    
    def __init__(self):
        self.skills_dir = get_abs_path("agent/skills")
        self.skills = {}
        self.skill_metadata = {}
    
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
                # 检查是否包含 SKILL.md 文件
                skill_md_path = os.path.join(item_path, "SKILL.md")
                script_path = os.path.join(item_path, "script", "main.py")
                
                if os.path.exists(skill_md_path) and os.path.exists(script_path):
                    skills.append(item)
                    logger.info(f"[SkillLoader] 发现技能: {item}")
        
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
            return {}
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析 SKILL.md 获取关键信息
            metadata = {
                "name": skill_name,
                "content": content
            }
            
            # 提取概览信息
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
            logger.error(f"[SkillLoader] 加载元数据失败: {e}")
            return {}
    
    def load_skill_function(self, skill_name: str):
        """
        加载技能的可执行函数
        :param skill_name: 技能名称
        :return: 技能函数
        """
        script_path = os.path.join(self.skills_dir, skill_name, "script", "main.py")
        
        if not os.path.exists(script_path):
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
            
            logger.warning(f"[SkillLoader] 未找到技能函数: {skill_name}")
            return None
        
        except Exception as e:
            logger.error(f"[SkillLoader] 加载技能失败: {e}", exc_info=True)
            return None
    
    def load_all_skills(self) -> Dict[str, Any]:
        """
        加载所有技能
        :return: 技能字典 {skill_name: (function, metadata)}
        """
        skills = {}
        skill_names = self.discover_skills()
        
        for name in skill_names:
            func = self.load_skill_function(name)
            metadata = self.load_skill_metadata(name)
            
            if func:
                skills[name] = {
                    "function": func,
                    "metadata": metadata
                }
                self.skill_metadata[name] = metadata
        
        logger.info(f"[SkillLoader] 共加载 {len(skills)} 个技能")
        return skills
    
    def get_tool_list(self) -> List:
        """
        获取所有技能的工具列表（用于 Agent）
        :return: 工具函数列表
        """
        skills = self.load_all_skills()
        return [skill["function"] for skill in skills.values()]
    
    def get_skill_descriptions(self) -> str:
        """
        获取所有技能的描述（用于提示词）
        :return: 格式化的技能描述字符串
        """
        descriptions = []
        
        for name, skill in self.skill_metadata.items():
            desc = skill.get("capability", "") or skill.get("overview", "")
            if desc:
                descriptions.append(f"- {name}: {desc}")
        
        return "\n".join(descriptions)


# 全局技能加载器
skill_loader = SkillLoader()
