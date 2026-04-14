"""
图片索引映射模块
记录 image_id → 文件路径映射，支持图片检索与引用
使用 pickle 存储，可迁移至 SQLite
"""

import os
import pickle
import uuid
from typing import Dict, Optional
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class ImageIndex:
    def __init__(self, index_path: str = None):
        self.index_path = index_path or get_abs_path("image_index.pkl")
        self.image_map: Dict[str, str] = {}  # image_id -> file_path
        self.path_to_id: Dict[str, str] = {}  # file_path -> image_id

    def add_image(self, file_path: str) -> str:
        """添加图片，返回 image_id"""
        if not os.path.exists(file_path):
            logger.error(f"图片文件不存在: {file_path}")
            return None

        if file_path in self.path_to_id:
            return self.path_to_id[file_path]

        image_id = str(uuid.uuid4())
        self.image_map[image_id] = file_path
        self.path_to_id[file_path] = image_id
        logger.info(f"图片已添加: {image_id} -> {file_path}")
        return image_id

    def get_path(self, image_id: str) -> Optional[str]:
        """根据 image_id 获取文件路径"""
        return self.image_map.get(image_id)

    def get_id(self, file_path: str) -> Optional[str]:
        """根据文件路径获取 image_id"""
        return self.path_to_id.get(file_path)

    def remove_image(self, image_id: str) -> bool:
        """移除图片"""
        if image_id not in self.image_map:
            return False
        file_path = self.image_map[image_id]
        del self.image_map[image_id]
        del self.path_to_id[file_path]
        logger.info(f"图片已移除: {image_id}")
        return True

    def list_images(self) -> Dict[str, str]:
        """列出所有图片映射"""
        return self.image_map.copy()

    def save(self):
        """保存索引到 pickle 文件"""
        data = {
            'image_map': self.image_map,
            'path_to_id': self.path_to_id
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"图片索引已保存到 {self.index_path}")

    def load(self):
        """从 pickle 文件加载索引"""
        if not os.path.exists(self.index_path):
            logger.warning(f"图片索引文件不存在: {self.index_path}")
            return
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
        self.image_map = data['image_map']
        self.path_to_id = data['path_to_id']
        logger.info(f"图片索引已从 {self.index_path} 加载")

    def get_metadata(self) -> Dict:
        """获取索引元数据"""
        return {
            'total_images': len(self.image_map),
            'index_path': self.index_path
        }


# 全局图片索引实例
image_index = ImageIndex()


if __name__ == "__main__":
    # 示例使用
    image_index.add_image("image/example.jpg")
    image_index.save()
    image_index.load()
    print(image_index.list_images())