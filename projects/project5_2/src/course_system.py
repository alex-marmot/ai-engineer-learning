import os
import sys
import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path to import extend
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import utils
try:
    from utils.redis_client import RedisManager
except ImportError:
    # Fallback
    sys.path.append(os.path.join(current_dir, '..'))
    try:
        from src.utils.redis_client import RedisManager
    except ImportError:
         logger.warning("Could not import RedisManager from src.utils or utils.")
         RedisManager = None

# Import extend modules
try:
    from extend.task_processor import TaskProcessor
    from extend.agent_executor import CourseAgentExecutor
    from extend.grpc_client import GrpcClient
    from extend.retry_manager import RetryManager
    from extend.state_synchronizer import StateSynchronizer
except ImportError as e:
    logger.error(f"Failed to import extend modules: {e}")
    raise

class CourseSystem:
    """
    课程生成系统 Worker 节点。
    编排任务获取、执行和结果上报，使用分布式组件。

    【设计对比注释】Project 5.1 vs Project 5.2
    ---------------------------------------------------------
    Project 5.1 (单体 CLI 模式):
    1. 交互方式: 用户通过命令行 (CLI) 直接交互，输入主题和要求。
    2. 任务调度: 硬编码的线性顺序执行 (Research -> Outline -> Chapter -> Review)，无队列概念。
    3. 状态存储: 使用本地 JSON 文件 (course_checkpoint.json) 保存会话状态。
    4. 架构: 单体应用，CourseSystem 类直接包含 CrewAI 的 Agent/Task 定义和执行逻辑。

    Project 5.2 (分布式 Worker 模式):
    1. 交互方式: 被动接收任务。系统作为 Worker 节点运行，不直接与用户交互。
    2. 任务调度: 基于 Redis 消息队列 (TaskProcessor)。
       - 生产者 (Producer) 将任务推送到 Redis 队列。
       - 消费者 (Worker) 从队列抢占式获取任务。
    3. 可靠性机制:
       - 分布式锁 (StateSynchronizer): 避免多节点重复执行同一任务。
       - 安全获取 (Safe Fetch): 使用 RPOPLPUSH 确保任务在处理完成前不会丢失。
       - 失败重试 (RetryManager): 遇到瞬时错误自动重试。
       - 心跳保活 (Heartbeat): 防止 Worker 宕机导致锁死。
    4. 架构: 分布式架构，模块化设计。
       - CourseSystem: 负责任务编排 (获取 -> 加锁 -> 执行 -> 上报)。
       - extend 模块: 封装了 Redis、gRPC、重试等通用分布式能力。
    ---------------------------------------------------------
    """

    def __init__(self, redis_host='localhost', redis_port=6379, grpc_target='localhost:50051'):
        load_dotenv()
        
        # 1. Initialize Redis (初始化 Redis 连接)
        try:
            self.redis_manager = RedisManager(host=redis_host, port=redis_port)
            logger.info(f"Redis connected: {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

        # 2. Initialize Modules (初始化各功能模块)
        # 任务处理模块：负责从 Redis 队列获取任务
        self.task_processor = TaskProcessor(self.redis_manager)
        # 状态同步模块：负责分布式锁和状态更新 (跨服务状态同步)
        self.state_synchronizer = StateSynchronizer(self.redis_manager)
        # gRPC 客户端：负责结果回传
        self.grpc_client = GrpcClient(target=grpc_target)
        # 重试管理器：负责失败重试策略
        self.retry_manager = RetryManager(max_retries=3) # Configurable (可配置)
        
        # 3. Initialize Agent Executor (初始化 Agent 执行器)
        try:
            self.agent_executor = CourseAgentExecutor()
            logger.info("Agent Executor initialized")
        except Exception as e:
            logger.error(f"Agent Executor initialization failed: {e}")
            raise

        self.worker_running = False

    def run(self):
        """
        入口方法：启动 Worker 循环。
        """
        self.start_worker()

    def start_worker(self, queues: List[str] = None):
        """
        启动 Worker 循环以获取并执行任务。
        
        Args:
            queues: 要监听的队列名称列表（例如：['tasks:high', 'tasks:default']）。如果为 None，则使用 TaskProcessor 的默认值。
        """
        self.worker_running = True
        target_queues = queues or self.task_processor.queues
        if queues:
            self.task_processor.queues = queues
            
        logger.info(f"Starting worker listening on {target_queues}")
        
        # Unique processing queue for this worker
        # 此 Worker 的唯一处理队列
        processing_queue = f"tasks:processing:{self.redis_manager.worker_id}"
        
        # 0. Resume interrupted tasks (处理遗留任务)
        self._process_leftovers(processing_queue)

        while self.worker_running:
            # 1. Fetch Task (Safe)
            # Use safe_fetch which returns (task_id, task_data, queue_name, raw_json)
            task_tuple = self.task_processor.safe_fetch(processing_queue)
            if not task_tuple:
                time.sleep(1) # Prevent busy wait
                continue

            task_id, task_data, queue_name, raw_json = task_tuple
            
            # Process the task
            self._handle_task(task_id, task_data, processing_queue, raw_json)

    def _process_leftovers(self, processing_queue: str):
        """
        处理上次运行遗留在处理队列中的任务。
        """
        logger.info(f"Checking for leftovers in {processing_queue}...")
        # Since we are single-threaded, there should be at most one item if we ack correctly.
        # But if we crashed multiple times or logic changed, maybe more.
        # We iterate until empty.
        
        while True:
            # Check the first item
            task_json = self.redis_manager.client.lindex(processing_queue, 0)
            if not task_json:
                break
            
            logger.info(f"Found leftover task in {processing_queue}")
            try:
                task_data = json.loads(task_json)
                task_id = task_data.get('id')
                if task_id:
                     # Re-use handle logic
                     self._handle_task(task_id, task_data, processing_queue, task_json)
                else:
                     logger.warning("Leftover task missing ID. Removing.")
                     self.task_processor.remove_from_processing(processing_queue, task_json)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in leftover. Removing.")
                self.task_processor.remove_from_processing(processing_queue, task_json)
            except Exception as e:
                logger.error(f"Error processing leftover: {e}")
                # If we fail to process, we might be stuck. 
                # For now, we will retry via _handle_task logic, but if that fails, 
                # it stays in processing?
                # If _handle_task returns, it means we are done (success or fail).
                # _handle_task MUST remove the task from processing queue finally.
                pass

    def _handle_task(self, task_id: str, task_data: Dict[str, Any], processing_queue: str, raw_json: str):
        """
        Common logic to process a task: Check Lock -> Check State -> Execute -> Report -> Ack
        通用任务处理逻辑：检查锁 -> 检查状态 -> 执行 -> 上报 -> 确认
        
        【设计对比】任务处理流程
        Project 5.1: 简单的函数调用，没有状态检查（依赖内存状态）和锁机制。
        Project 5.2: 健壮的分布式处理流程：
        1. 幂等性检查 (Idempotency): 检查任务是否已完成，避免重复消费。
        2. 分布式锁 (Distributed Lock): 确保同一时刻只有一个 Worker 处理该任务。
        3. 状态同步 (State Sync): 将任务状态 (running/completed/failed) 实时同步到 Redis，供前端或监控查询。
        4. 结果上报 (Result Reporting): 通过 gRPC 将结果回传给主服务。
        5. 消息确认 (ACK): 处理完成后从 Redis 队列移除消息。
        """
        try:
            # 2. Check if already completed (Idempotency Phase 1)
            # 检查是否已完成 (幂等性阶段 1)
            state = self.state_synchronizer.get_state(task_id)
            if state and state.get('status') == 'completed':
                logger.info(f"Task {task_id} already completed. Skipping and Acking.")
                self.task_processor.remove_from_processing(processing_queue, raw_json)
                return

            # 3. Idempotency Check & Locking (幂等性检查与加锁)
            if not self.state_synchronizer.acquire_lock(task_id, self.redis_manager.worker_id):
                logger.warning(f"Task {task_id} is locked by another worker. Skipping.")
                # 如果已被其他 Worker 加锁，我们是否应该把它从自己的 processing 队列里移除？
                # 注意：我们通过 safe_fetch 拿到任务，所以它目前位于本 Worker 专属的 processing 队列。
                # 若另一 Worker 已持有锁，说明它正在处理该任务。
                # 它怎么拿到任务的？除非我们之前崩溃，它从别的副本队列里取到。
                # 或者：源队列里存在重复消息？
                # 若源队列重复，我们各自拿到一份副本，谁先抢到锁谁执行。
                # 抢锁失败的一方应当丢弃（Ack）并相信对方会完成。
                logger.info(f"Acking duplicate task {task_id} (locked by other).")
                self.task_processor.remove_from_processing(processing_queue, raw_json)
                return

            # 4. Heartbeat (心跳保活)
            stop_heartbeat = threading.Event()
            hb_thread = threading.Thread(target=self._heartbeat_loop, args=(task_id, stop_heartbeat))
            hb_thread.start()

            try:
                logger.info(f"Processing task {task_id}")
                # 更新状态为运行中 (跨服务状态同步)
                self.state_synchronizer.sync_state(task_id, "running")

                # 5. Execute with Retry (带重试机制的执行)
                # 调用 Agent 执行业务逻辑
                result = self._process_task_with_retry(task_id, task_data)

                # 6. Report Success (上报成功结果)
                # 更新状态为已完成，并将结果保存到 Redis (方便客户端获取)
                result_json = json.dumps(result, ensure_ascii=False)
                self.state_synchronizer.sync_state(task_id, "completed", data={"result": result_json})
                
                # 通过 gRPC 回传结果
                self.grpc_client.report_result(task_id, "SUCCESS", result)
                logger.info(f"Task {task_id} completed successfully.")
                logger.info(f"Task Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                # 更新状态为失败，并记录错误信息
                self.state_synchronizer.sync_state(task_id, "failed", data={"error": str(e)})
                # 通过 gRPC 上报失败
                self.grpc_client.report_result(task_id, "FAILURE", {"error": str(e)})
                
            finally:
                # 停止心跳并释放锁
                stop_heartbeat.set()
                hb_thread.join()
                self.state_synchronizer.release_lock(task_id, self.redis_manager.worker_id)
                
                # 7. 确认（ACK）：无论成功或失败，都从处理队列中移除
                # 注意：如果执行失败，我们直接移除；若希望实现全局重试（重新入队），
                # 应在此处将任务重新推回原队列或死信队列（DLQ）。
                # 当前需求：“支持失败重试机制”。
                # 我们已在 _process_task_with_retry 中实现本地重试；
                # 若本地重试耗尽仍失败，则视为最终失败。
                self.task_processor.remove_from_processing(processing_queue, raw_json)

        except Exception as e:
            logger.error(f"Unexpected error in _handle_task: {e}")
            # Ensure we don't block forever if something weird happens
            # But maybe we shouldn't ack if it was a transient system error?
            # For safety, let's keep it in processing if it was a system crash (unhandled).
            # But here we caught Exception.
            pass

    def _process_task_with_retry(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task using AgentExecutor with RetryManager.
        使用 AgentExecutor 执行任务，并结合 RetryManager 进行重试。
        
        【设计对比】错误处理与重试
        Project 5.1: 交互式重试。出错时通过 print/input 询问用户 "是否重试? (y/n)"。
        Project 5.2: 自动化重试机制 (RetryManager)。
        - 策略: 捕获异常后，根据配置的最大重试次数 (max_retries) 决定是否重试。
        - 退避: 使用指数退避算法 (Exponential Backoff)，重试间隔随次数增加，减轻系统负载。
        - 记录: 详细记录每次重试的原因和上下文，便于排查。
        """
        retries = 0
        while True:
            try:
                # Delegate to AgentExecutor (委托给 AgentExecutor 执行)
                # 任务数据应包含 'phase' 和 AgentExecutor 所需的其他参数。
                return self.agent_executor.execute(task_data)
            except Exception as e:
                # Check if we should retry (检查是否需要重试)
                if self.retry_manager.should_retry(retries):
                    self.retry_manager.log_failure(task_id, e, context=f"Attempt {retries+1}")
                    # Wait for backoff time (指数退避等待)
                    self.retry_manager.wait_for_retry(retries)
                    retries += 1
                    # Update state to 'retrying' (更新状态为重试中)
                    self.state_synchronizer.sync_state(task_id, "retrying", data={"retry_count": retries})
                else:
                    raise e

    def _heartbeat_loop(self, task_id, stop_event):
        """Keep the task lock alive. (保持任务锁活跃)"""
        while not stop_event.is_set():
            self.redis_manager.heartbeat(task_id)
            # Use wait instead of sleep to allow immediate exit
            if stop_event.wait(10):
                break

    def submit_task(self, task_data: Dict[str, Any], queue_name='course_tasks:default'):
        """
        提交任务到 Redis 的辅助方法 (生产者模式)。
        """
        import uuid
        if 'id' not in task_data:
            task_data['id'] = str(uuid.uuid4())
        
        # Ensure it has a phase if not provided, though execute expects it
        if 'phase' not in task_data:
            logger.warning("Task submitted without 'phase'. execution might fail.")

        self.redis_manager.client.rpush(queue_name, json.dumps(task_data))
        logger.info(f"Task {task_data['id']} submitted to {queue_name}")
        return task_data['id']
