from typing import List, Dict, Any, Optional
from utils.logger_handler import logger


class PathPlanner:
    """
    路径规划模块：为子任务排序，确定执行逻辑和优先级，并动态调整
    """
    
    PRIORITY_WEIGHTS = {
        "high": 3,
        "medium": 2,
        "low": 1
    }
    
    def __init__(self):
        self.execution_history = []
    
    def plan(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        规划执行路径
        
        :param tasks: 子任务列表
        :return: 排序后的执行路径
        """
        if not tasks:
            logger.warning("[PathPlanner] 任务列表为空")
            return []
        
        # 构建依赖关系图
        task_map = {task["id"]: task for task in tasks}
        
        # 检测循环依赖
        if self._has_circular_dependency(task_map):
            logger.warning("[PathPlanner] 检测到循环依赖，使用拓扑排序")
            return self._topological_sort(task_map)
        
        # 基于优先级和依赖关系排序
        sorted_tasks = self._sort_by_priority_and_dependencies(task_map)
        
        logger.info(f"[PathPlanner] 路径规划完成，共 {len(sorted_tasks)} 个任务")
        return sorted_tasks
    
    def _sort_by_priority_and_dependencies(self, task_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于优先级和依赖关系排序"""
        result = []
        completed = set()
        
        while len(result) < len(task_map):
            ready_tasks = []
            
            for task_id, task in task_map.items():
                if task_id in completed:
                    continue
                
                # 检查依赖是否都已完成
                dependencies = task.get("dependencies", [])
                if all(dep in completed for dep in dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # 无法继续，可能存在循环依赖
                logger.warning("[PathPlanner] 无法继续规划，可能存在循环依赖")
                break
            
            # 按优先级排序
            ready_tasks.sort(key=lambda t: self.PRIORITY_WEIGHTS.get(t.get("priority", "medium"), 2), reverse=True)
            
            # 添加最高优先级的任务
            next_task = ready_tasks[0]
            result.append(next_task)
            completed.add(next_task["id"])
        
        return result
    
    def _topological_sort(self, task_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """拓扑排序处理循环依赖"""
        in_degree = {}
        adjacency = {}
        
        for task_id, task in task_map.items():
            in_degree[task_id] = 0
            adjacency[task_id] = []
        
        for task_id, task in task_map.items():
            dependencies = task.get("dependencies", [])
            for dep in dependencies:
                if dep in adjacency:
                    adjacency[dep].append(task_id)
                    in_degree[task_id] += 1
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(task_map[current])
            
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _has_circular_dependency(self, task_map: Dict[str, Dict[str, Any]]) -> bool:
        """检测循环依赖"""
        def dfs(node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            dependencies = task_map.get(node, {}).get("dependencies", [])
            for dep in dependencies:
                if dep not in visited:
                    if dfs(dep, visited, rec_stack):
                        return True
                elif rec_stack.get(dep, False):
                    return True
            
            rec_stack[node] = False
            return False
        
        visited = {}
        rec_stack = {}
        
        for task_id in task_map:
            if task_id not in visited:
                if dfs(task_id, visited, rec_stack):
                    return True
        
        return False
    
    def record_execution(self, task_id: str, success: bool, result: Optional[str] = None):
        """记录任务执行结果"""
        self.execution_history.append({
            "task_id": task_id,
            "success": success,
            "result": result,
            "timestamp": self._get_timestamp()
        })
        logger.info(f"[PathPlanner] 任务执行记录: {task_id} - {'成功' if success else '失败'}")
    
    def suggest_alternative(self, failed_task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """为失败的任务建议备选方案"""
        alternatives = {
            "rag_summarize": {
                "title": "尝试不同检索词",
                "description": "使用不同的检索词重新检索知识库",
                "tool": "rag_summarize"
            },
            "default": {
                "title": "跳过此步骤",
                "description": "跳过当前步骤，继续执行后续任务",
                "tool": None
            }
        }
        
        tool = failed_task.get("tool")
        return alternatives.get(tool, alternatives["default"])
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


# 全局实例
path_planner = PathPlanner()
