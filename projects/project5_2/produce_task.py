import json
import uuid
import sys
import os
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# 确保导入路径正确，将 src 目录添加到系统路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils.redis_client import RedisManager

class CourseGenerationClient:
    """
    课程生成客户端类
    负责与 Redis 交互，协调课程生成的各个阶段（调研、大纲、章节编写、审核）。
    """
    def __init__(self):
        """
        初始化客户端
        加载环境变量，连接 Redis，并设置任务队列名称。
        """
        load_dotenv()
        try:
            self.redis = RedisManager()
            print("[Init] Connected to Redis. (Redis 连接成功)")
        except Exception as e:
            print(f"[Error] Failed to connect to Redis: {e} (Redis 连接失败)")
            sys.exit(1)
            
        # 定义 Redis 任务队列名称
        self.queue_name = "tasks:default"

    def submit_task_and_wait(self, payload: Dict[str, Any], timeout: int = 600) -> Optional[Dict[str, Any]]:
        """
        提交任务到 Redis 并等待完成。
        
        Args:
            payload: 任务数据字典
            timeout: 超时时间（秒），默认 600 秒
            
        Returns:
            执行结果字典，如果失败或超时则返回 None
        """
        # 生成或获取任务 ID
        task_id = payload.get('id', str(uuid.uuid4()))
        payload['id'] = task_id
        payload['timestamp'] = int(time.time())
        
        print(f"\n[Task] Submitting '{payload['phase']}' task (ID: {task_id})... (正在提交任务...)")
        
        try:
            # 将任务推送到 Redis 队列
            self.redis.client.rpush(self.queue_name, json.dumps(payload))
        except Exception as e:
            print(f"[Error] Failed to push task: {e} (任务提交失败)")
            return None
            
        # 轮询检查结果
        start_time = time.time()
        print(f"[Wait] Waiting for result... (等待结果...)", end="", flush=True)
        
        while True:
            # 检查超时
            if time.time() - start_time > timeout:
                print("\n[Timeout] Task took too long. (任务超时)")
                return None
                
            # 获取任务状态
            state_key = f"task:{task_id}:state"
            state = self.redis.client.hgetall(state_key)
            
            if state:
                status = state.get('status')
                # 任务完成
                if status == 'completed':
                    print(f" Done! ({time.time() - start_time:.1f}s)")
                    
                    # 解析结果
                    result_raw = state.get('result')
                    if result_raw:
                        try:
                            return json.loads(result_raw)
                        except json.JSONDecodeError:
                            print(f"\n[Error] Invalid JSON in result: {result_raw} (结果 JSON 格式错误)")
                            return None
                    else:
                        print("\n[Warning] Task completed but no result data found. (任务完成但无结果数据)")
                        return None
                        
                # 任务失败
                elif status == 'failed':
                    print(f"\n[Failure] Task failed. Error: {state.get('extra_info')} (任务失败)")
                    return None
            
            # 等待 1 秒后重试
            time.sleep(1)
            print(".", end="", flush=True)

    def run(self):
        """
        运行课程生成主流程
        包括：输入信息 -> 市场调研 -> 大纲制定 -> 章节编写 -> 全文审核
        """
        print("="*50)
        print("AI Course Generation Client (Interactive)")
        print("AI 课程生成客户端 (交互式)")
        print("="*50)
        
        # 1. 输入基本信息 (Input Basic Info)
        topic = input("请输入课程主题 (Topic): ").strip()
        if not topic: topic = "Python 异步编程" # 默认值
        
        requirements = input("请输入课程要求 (Requirements): ").strip()
        if not requirements: requirements = "适合初学者" # 默认值
        
        print(f"\nTopic: {topic}")
        print(f"Requirements: {requirements}")
        
        # 2. 市场调研阶段 (Research Phase)
        chosen_direction = self._run_research(topic, requirements)
        if not chosen_direction: return
        
        # 3. 大纲制定阶段 (Outline Phase)
        outline_data = self._run_outline(topic, requirements, chosen_direction)
        if not outline_data: return
        
        # 4. 章节编写阶段 (Chapter Generation Phase)
        course_content = self._run_chapters(topic, requirements, outline_data)
        if not course_content: return
        
        # 5. 审核阶段 (Review Phase)
        self._run_review(topic, requirements, chosen_direction, outline_data['course_title'], course_content)
        
        print("\n" + "="*50)
        print("🎉 课程生成流程全部完成！")
        print("="*50)

    def _run_research(self, topic: str, requirements: str) -> Optional[str]:
        """
        执行市场调研阶段
        """
        print("\n" + "-"*30)
        print("Phase 1: Research (市场调研)")
        print("-"*30)
        
        payload = {
            "type": "course_generation",
            "phase": "research",
            "topic": topic,
            "requirements": requirements
        }
        
        result = self.submit_task_and_wait(payload)
        if not result: return None
        
        # 提取建议
        suggestions = result.get('result', '')
        print("\n📋 建议的课程方向:")
        print(suggestions)
        
        print("\n" + "-"*30)
        choice = input("请输入您选择的方向 (复制粘贴或简要描述): ").strip()
        # 如果用户未输入，默认使用第一行建议（这里逻辑可以根据需要调整，目前是回退到用户必须输入或使用默认）
        return choice if choice else suggestions.split('\n')[0]

    def _run_outline(self, topic: str, requirements: str, chosen_direction: str) -> Optional[Dict]:
        """
        执行大纲制定阶段
        """
        print("\n" + "-"*30)
        print("Phase 2: Outline (大纲制定)")
        print("-"*30)
        
        while True:
            payload = {
                "type": "course_generation",
                "phase": "outline",
                "topic": topic,
                "requirements": requirements,
                "chosen_direction": chosen_direction
            }
            
            result = self.submit_task_and_wait(payload)
            if not result: return None
            
            outline_data = result.get('result')
            # 验证大纲格式
            if not outline_data or not isinstance(outline_data, dict) or 'chapters' not in outline_data:
                print("\n[Error] Invalid outline format received. (收到的大纲格式无效)")
                print(outline_data)
                retry = input("Retry? (y/n): ").strip().lower()
                if retry == 'y': continue
                return None
                
            print("\n📋 生成的课程大纲:")
            chapters = outline_data.get('chapters', [])
            for i, chapter in enumerate(chapters):
                print(f"第 {i+1} 章: {chapter['title']} - {chapter['summary']}")
            
            print("\n")
            action = input("[1] 确认继续  [2] 修改要求重试  [3] 退出 : ").strip()
            if action == '1':
                if 'course_title' not in outline_data:
                    outline_data['course_title'] = chosen_direction
                return outline_data
            elif action == '2':
                req_update = input("请输入修改建议: ").strip()
                requirements += f" (修改建议: {req_update})"
                continue
            else:
                return None

    def _run_chapters(self, topic: str, requirements: str, outline_data: Dict) -> Optional[List[str]]:
        """
        执行章节编写阶段
        """
        print("\n" + "-"*30)
        print("Phase 3: Chapter Writing (章节编写)")
        print("-"*30)
        
        chapters = outline_data.get('chapters', [])
        course_title = outline_data.get('course_title', 'Unknown Course')
        course_content = []
        
        for i, chapter in enumerate(chapters):
            print(f"\nProcessing Chapter {i+1}/{len(chapters)}: {chapter['title']}")
            
            current_reqs = requirements
            while True:
                payload = {
                    "type": "course_generation",
                    "phase": "chapter",
                    "topic": topic,
                    "requirements": current_reqs,
                    "chapter_index": i+1,
                    "chapter_title": chapter['title'],
                    "chapter_summary": chapter['summary'],
                    "course_title": course_title
                }
                
                result = self.submit_task_and_wait(payload)
                if not result: return None
                
                content = result.get('result', '')
                
                print(f"\n📄 第 {i+1} 章内容预览 (前500字):")
                print("-" * 20)
                print(content[:500] + "...")
                print("-" * 20)
                
                action = input(f"\n[1] 确认本章  [2] 修改重写  [3] 退出 : ").strip()
                if action == '1':
                    course_content.append(f"# 第 {i+1} 章: {chapter['title']}\n\n{content}")
                    break
                elif action == '2':
                    feedback = input("请输入修改建议: ").strip()
                    current_reqs += f" (本章修改建议: {feedback})"
                    continue
                else:
                    return None
                    
        return course_content

    def _run_review(self, topic: str, requirements: str, chosen_direction: str, course_title: str, course_content: List[str]):
        """
        执行全文审核阶段
        """
        print("\n" + "-"*30)
        print("Phase 4: Final Review (全文审核)")
        print("-"*30)
        
        full_content = "\n\n".join(course_content)
        
        payload = {
            "type": "course_generation",
            "phase": "review",
            "topic": topic,
            "requirements": requirements,
            "chosen_direction": chosen_direction,
            "course_title": course_title,
            "course_content": full_content
        }
        
        result = self.submit_task_and_wait(payload)
        if result:
            report = result.get('result', '')
            print("\n📋 最终审核报告:")
            print(report)
            
            # 保存到文件
            filename = f"course_output_{int(time.time())}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {course_title}\n\n")
                f.write(full_content)
                f.write("\n\n---\n# 审核报告\n\n")
                f.write(report)
            print(f"\n[File] 课程内容已保存至: {filename}")

if __name__ == "__main__":
    client = CourseGenerationClient()
    client.run()
