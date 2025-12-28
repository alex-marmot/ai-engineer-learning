import grpc
import logging
import json
import time
import sys
import os
from typing import Dict, Any, Generator

# 确保可以从 src 导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # project5_2
src_path = os.path.join(project_root, 'src')
protos_path = os.path.join(src_path, 'protos')

if src_path not in sys.path:
    sys.path.append(src_path)
if protos_path not in sys.path:
    sys.path.append(protos_path)

try:
    import task_pb2
    import task_pb2_grpc
except ImportError as e:
    logging.warning(f"导入 protos 失败: {e}。gRPC 功能将受限。")
    task_pb2 = None
    task_pb2_grpc = None

logger = logging.getLogger(__name__)

class GrpcClient:
    """
    封装 gRPC 通信以用于上报任务结果。
    """
    def __init__(self, target: str = 'localhost:50051'):
        self.target = target
        self.channel = None
        self.stub = None
        self._connect()

    def _connect(self):
        """初始化 gRPC 通道和存根。"""
        if task_pb2_grpc:
            try:
                self.channel = grpc.insecure_channel(self.target)
                self.stub = task_pb2_grpc.TaskServiceStub(self.channel)
                logger.info(f"gRPC 客户端已连接到 {self.target}")
            except Exception as e:
                logger.error(f"连接到 gRPC 服务器失败: {e}")

    def report_result(self, task_id: str, status: str, result_data: Dict[str, Any]) -> bool:
        """
        将执行结果上报回服务器。
        """
        if not self.stub:
            logger.error("gRPC 存根不可用。无法上报结果。")
            return False

        try:
            # 将状态字符串映射到枚举
            status_enum = getattr(task_pb2.TaskResult.Status, status.upper(), task_pb2.TaskResult.Status.FAILURE)
            
            result = task_pb2.TaskResult(
                task_id=task_id,
                status=status_enum,
                result_data_json=json.dumps(result_data, ensure_ascii=False),
                timestamp=str(time.time())
            )
            
            # 流式请求 (根据 proto 定义: stream TaskResult)
            def request_generator():
                yield result

            responses = self.stub.ReportResult(request_generator())
            
            # 检查响应
            for resp in responses:
                if resp.task_id == task_id and resp.received:
                    logger.info(f"任务 {task_id} 结果上报成功。")
                    return True
            
            return False

        except grpc.RpcError as e:
            logger.error(f"任务 {task_id} 的 gRPC RPC 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"上报任务 {task_id} 结果时出错: {e}")
            return False

    def close(self):
        """关闭 gRPC 通道。"""
        if self.channel:
            self.channel.close()
