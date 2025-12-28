import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retry_manager import RetryManager
from task_processor import TaskProcessor
from state_synchronizer import StateSynchronizer
from grpc_client import GrpcClient

class TestRetryManager(unittest.TestCase):
    def test_should_retry(self):
        rm = RetryManager(max_retries=3)
        self.assertTrue(rm.should_retry(0))
        self.assertTrue(rm.should_retry(2))
        self.assertFalse(rm.should_retry(3))

    def test_backoff(self):
        rm = RetryManager(base_delay=1.0, max_delay=10.0)
        # 1 * 2^0 = 1.0 (plus jitter)
        self.assertGreaterEqual(rm.get_backoff_time(0), 1.0)
        # 1 * 2^2 = 4.0
        self.assertGreaterEqual(rm.get_backoff_time(2), 4.0)

class TestTaskProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MagicMock()
        self.processor = TaskProcessor(self.mock_redis)

    def test_fetch_task_success(self):
        task_data = {"id": "123", "type": "test", "payload": {}}
        self.mock_redis.client.blpop.return_value = ('tasks:high', json.dumps(task_data))
        
        tid, data, queue = self.processor.fetch_task()
        self.assertEqual(tid, "123")
        self.assertEqual(queue, "tasks:high")

    def test_fetch_task_empty(self):
        self.mock_redis.client.blpop.return_value = None
        result = self.processor.fetch_task()
        self.assertIsNone(result)

    def test_parse_task_valid(self):
        task = {"id": "1", "type": "test"}
        self.assertEqual(self.processor.parse_task(task), task)

    def test_parse_task_invalid(self):
        task = {"type": "test"} # Missing id
        with self.assertRaises(ValueError):
            self.processor.parse_task(task)

class TestStateSynchronizer(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MagicMock()
        self.sync = StateSynchronizer(self.mock_redis)

    def test_acquire_lock(self):
        self.mock_redis.client.set.return_value = True
        self.assertTrue(self.sync.acquire_lock("res1", "owner1"))
        self.mock_redis.client.set.assert_called_with("lock:res1", "owner1", ex=300, nx=True)

    def test_release_lock(self):
        self.mock_redis.client.get.return_value = "owner1"
        self.sync.release_lock("res1", "owner1")
        self.mock_redis.client.delete.assert_called_with("lock:res1")

class TestGrpcClient(unittest.TestCase):
    @patch('grpc_client.task_pb2_grpc')
    @patch('grpc_client.grpc')
    def test_report_result(self, mock_grpc, mock_pb2_grpc):
        # Setup mocks
        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel
        mock_pb2_grpc.TaskServiceStub.return_value = mock_stub
        
        client = GrpcClient()
        
        # Mock response
        mock_resp = MagicMock()
        mock_resp.task_id = "123"
        mock_resp.received = True
        mock_stub.ReportResult.return_value = [mock_resp]
        
        result = client.report_result("123", "SUCCESS", {})
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
