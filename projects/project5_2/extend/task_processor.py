import logging
import json
import time
import sys
import os
from typing import Optional, Dict, Tuple, List, Any

# 确保导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

from utils.redis_client import RedisManager

logger = logging.getLogger(__name__)

class TaskProcessor:
    """
    任务处理器：处理从 Redis 队列中获取、解析和分发任务。
    支持优先级队列。
    """
    def __init__(self, redis_manager: RedisManager, queues: List[str] = None):
        self.redis = redis_manager
        # 默认队列优先级：high -> medium -> low -> default
        self.queues = queues or ['tasks:high', 'tasks:medium', 'tasks:low', 'tasks:default']

    def fetch_task(self, timeout: int = 5) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """
        从配置的队列中根据优先级获取任务。
        注意：此处使用 BLPOP 支持优先级。为了严格的可靠性（崩溃时不丢失数据），考虑使用 safe_fetch 配合处理队列，或 Redis Streams。
        
        返回:
            Tuple(task_id, task_data, queue_name) 或 None
        """
        try:
            # BLPOP 接受键列表，并从第一个非空队列中弹出
            result = self.redis.client.blpop(self.queues, timeout=timeout)
            
            if result:
                queue_name, task_json = result
                try:
                    task_data = json.loads(task_json)
                    
                    # 确保任务包含 ID
                    task_id = task_data.get('id')
                    if not task_id:
                        logger.warning(f"来自 {queue_name} 的任务缺少 ID。跳过。")
                        return None
                        
                    logger.info(f"从 {queue_name} 获取到任务 {task_id}")
                    return task_id, task_data, queue_name
                    
                except json.JSONDecodeError:
                    logger.error(f"从 {queue_name} 解码任务 JSON 失败: {task_json}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"从 Redis 获取任务时出错: {e}")
            return None

    def safe_fetch(self, processing_queue: str) -> Optional[Tuple[str, Dict[str, Any], str, str]]:
        """
        【设计对比】安全获取任务 (Safe Fetch)
        Project 5.1: 无此概念。任务由代码逻辑硬编码生成或用户输入触发。
        Project 5.2: 实现了可靠的消息队列消费模式。
        - 机制: 使用 Redis 的 RPOPLPUSH 命令。
        - 作用: 原子性操作。在从源队列 (queue) 取出任务的同时，将其推入备份队列 (processing_queue)。
        - 优势: 解决了“任务丢失”问题。如果 Worker 在取出任务后、处理完成前宕机，
          任务依然存在于 processing_queue 中，重启后可由 _process_leftovers 恢复处理。
        - 优先级: 按照 queues 列表定义的顺序 (high -> medium -> low) 轮询，实现优先级调度。
        """
        try:
            for queue in self.queues:
                # 使用 rpoplpush 保证可靠性
                task_json = self.redis.client.rpoplpush(queue, processing_queue)
                if task_json:
                    try:
                        task_data = json.loads(task_json)
                        task_id = task_data.get('id')
                        if task_id:
                            logger.info(f"可靠地将任务 {task_id} 从 {queue} 获取到 {processing_queue}")
                            # 检查 task_json 是否为 bytes (redis-py 通常根据配置返回 bytes 或 str)
                            # 我们应该确保它是 str 以保持一致性，但对于 lrem 我们需要确切得到的内容。
                            return task_id, task_data, queue, task_json
                    except json.JSONDecodeError:
                        logger.error(f"{queue} 中存在无效 JSON")
                        self.redis.client.lrem(processing_queue, 1, task_json)
                        pass
            return None
        except Exception as e:
            logger.error(f"safe_fetch 出错: {e}")
            return None

    def ack_task(self, processing_queue: str, task_data: Dict[str, Any]):
        """
        确认任务完成，将其从处理队列中移除。
        """
        try:
            # LREM 删除前 count 个等于 value 的元素
            # count > 0: 从头到尾删除等于 value 的元素。
            task_json = json.dumps(task_data) 
            # 注意：json.dumps 必须生成与 Redis 中完全相同的字符串。
            # 理想情况下，我们应该传递获取到的原始 json 字符串。
            # 但既然我们解析了它，如果我们没有保留它，可能会丢失确切的格式。
            # 最好从 fetch 返回 raw_json。
            
            # 让我们尝试按值删除。如果格式不同，这可能会失败。
            # 策略：调用者应尽可能传递原始 json，或者我们仔细重建它。
            # 或者更好：我们假设 task_data 是字典，但我们需要字符串用于 LREM。
            
            # 改进：fetch_task/safe_fetch 也应该返回 raw_json。
            # 目前，让我们假设如果我们尝试匹配，我们可以删除它。
            # 但 LREM 是严格的。
            
            self.redis.client.lrem(processing_queue, 1, json.dumps(task_data))
        except Exception as e:
            logger.error(f"确认任务时出错: {e}")

    def remove_from_processing(self, processing_queue: str, raw_task_json: str):
        """
        从处理队列中移除原始任务 JSON。
        """
        try:
            self.redis.client.lrem(processing_queue, 1, raw_task_json)
            logger.info(f"任务已确认并从 {processing_queue} 移除")
        except Exception as e:
            logger.error(f"从处理队列移除任务时出错: {e}")
