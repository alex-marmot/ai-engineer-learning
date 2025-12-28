import redis
import logging
import json
import time
import uuid
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password, 
            decode_responses=True
        )
        self.worker_id = str(uuid.uuid4())
        logger.info(f"RedisManager initialized with Worker ID: {self.worker_id}")

    def fetch_task(self, queue_name: str, timeout: int = 0) -> Optional[Tuple[str, Dict]]:
        """
        Fetch a task from the Redis List using BLPOP.
        Returns (task_id, task_data_dict) or None.
        使用 BLPOP 从 Redis List 获取任务。
        返回 (task_id, task_data_dict) 或 None。
        """
        try:
            # BLPOP returns (key, value)
            result = self.client.blpop(queue_name, timeout=timeout)
            if result:
                _, task_json = result
                try:
                    task_data = json.loads(task_json)
                    # Assume task_data has 'id' or we generate one?
                    # The prompt says "Task ID (UUID format)" is in gRPC result.
                    # It's better if the task payload has the ID.
                    task_id = task_data.get('id') or str(uuid.uuid4())
                    return task_id, task_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode task JSON: {task_json}")
                    return None
            return None
        except redis.RedisError as e:
            logger.error(f"Redis error fetching task: {e}")
            return None

    def acquire_lock(self, task_id: str, expire_seconds: int = 300) -> bool:
        """
        Acquire a distributed lock for the task to ensure single execution.
        为任务获取分布式锁以确保单次执行。
        """
        lock_key = f"lock:task:{task_id}"
        try:
            # setnx with expiry
            # 使用 setnx 设置过期时间
            acquired = self.client.set(lock_key, self.worker_id, ex=expire_seconds, nx=True)
            return bool(acquired)
        except redis.RedisError as e:
            logger.error(f"Error acquiring lock for {task_id}: {e}")
            return False

    def release_lock(self, task_id: str):
        """Release the lock if held by this worker. (如果此 worker 持有锁，则释放锁)"""
        lock_key = f"lock:task:{task_id}"
        try:
            # Simple delete for now. Ideally check ownership with Lua script.
            val = self.client.get(lock_key)
            if val == self.worker_id:
                self.client.delete(lock_key)
        except redis.RedisError as e:
            logger.error(f"Error releasing lock for {task_id}: {e}")

    def update_state(self, task_id: str, status: str, retry_count: int = 0, extra_info: Dict = None):
        """
        Update task state in Redis Hash.
        在 Redis Hash 中更新任务状态。
        """
        key = f"task:{task_id}:state"
        mapping = {
            "status": status,
            "last_update": time.time(),
            "retry_count": retry_count,
            "worker_id": self.worker_id
        }
        if extra_info:
            mapping.update(extra_info)
        
        try:
            self.client.hset(key, mapping=mapping)
            # Publish event
            # 发布事件
            self.client.publish("task_status_updates", json.dumps({
                "task_id": task_id,
                "status": status,
                "timestamp": time.time()
            }))
        except redis.RedisError as e:
            logger.error(f"Error updating state for {task_id}: {e}")

    def heartbeat(self, task_id: str):
        """Update last_update time for a running task. (更新正在运行任务的 last_update 时间)"""
        if task_id:
            key = f"task:{task_id}:state"
            try:
                self.client.hset(key, "last_update", time.time())
                # Also refresh lock
                # 同时刷新锁
                self.client.expire(f"lock:task:{task_id}", 300)
            except redis.RedisError as e:
                logger.error(f"Heartbeat error for {task_id}: {e}")

    def close(self):
        self.client.close()
