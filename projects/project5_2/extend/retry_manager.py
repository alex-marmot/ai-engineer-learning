import time
import logging
import random

logger = logging.getLogger(__name__)

class RetryManager:
    """
    使用指数退避策略管理重试逻辑。
    """
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def should_retry(self, current_retries: int) -> bool:
        """根据当前重试次数检查是否应该重试。"""
        return current_retries < self.max_retries

    def get_backoff_time(self, current_retries: int) -> float:
        """计算带有抖动的指数退避时间。"""
        delay = self.base_delay * (2 ** current_retries)
        # 添加抖动 (防止惊群效应)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, self.max_delay)

    def wait_for_retry(self, current_retries: int):
        """休眠计算出的退避时间。"""
        delay = self.get_backoff_time(current_retries)
        logger.info(f"重试尝试 {current_retries + 1}/{self.max_retries}. 等待 {delay:.2f}秒...")
        time.sleep(delay)

    def log_failure(self, task_id: str, error: Exception, context: str = ""):
        """记录失败详情。"""
        logger.error(f"任务 {task_id} 失败。上下文: {context}。错误: {error}")
