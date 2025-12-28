import logging
import json
import time
import sys
import os
from typing import Dict, Any, Optional

# 确保可以从 src 导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from utils.redis_client import RedisManager
except ImportError:
    # 如果导入失败，使用 Mock 进行开发
    logging.warning("无法导入 RedisManager。使用 Mock。")
    class RedisManager:
        def __init__(self, host, port): pass
        def update_state(self, *args, **kwargs): pass
        def acquire_lock(self, *args, **kwargs): return True
        def release_lock(self, *args, **kwargs): pass
        def client(self): pass

logger = logging.getLogger(__name__)

class StateSynchronizer:
    """
    使用 Redis 处理分布式状态同步和锁。
    """
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager

    def acquire_lock(self, resource_id: str, owner_id: str, expire_seconds: int = 300) -> bool:
        """
        获取资源的分布式锁。
        """
        key = f"lock:{resource_id}"
        try:
            # 使用 Redis SETNX 逻辑 (如果不存在则设置)
            # RedisManager.acquire_lock 使用内部 worker_id，这里我们允许自定义 owner_id
            acquired = self.redis.client.set(key, owner_id, ex=expire_seconds, nx=True)
            return bool(acquired)
        except Exception as e:
            logger.error(f"获取 {resource_id} 的锁时出错: {e}")
            return False

    def release_lock(self, resource_id: str, owner_id: str):
        """
        如果持有者是 owner_id，则释放锁。
        """
        key = f"lock:{resource_id}"
        try:
            # 简单的检查并删除。为了严格安全，应使用 Lua 脚本。
            val = self.redis.client.get(key)
            if val == owner_id:
                self.redis.client.delete(key)
        except Exception as e:
            logger.error(f"释放 {resource_id} 的锁时出错: {e}")

    def sync_state(self, task_id: str, status: str, data: Dict[str, Any] = None):
        """
        将任务状态同步到 Redis，以便跨服务可见。
        """
        try:
            self.redis.update_state(task_id, status, extra_info=data)
            logger.debug(f"任务 {task_id} 状态已同步: {status}")
        except Exception as e:
            logger.error(f"同步任务 {task_id} 状态失败: {e}")

    def get_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        检索任务的当前状态。
        """
        key = f"task:{task_id}:state"
        try:
            return self.redis.client.hgetall(key)
        except Exception as e:
            logger.error(f"获取任务 {task_id} 状态失败: {e}")
            return None
