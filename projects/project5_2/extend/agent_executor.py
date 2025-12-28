import os
import sys
import yaml
import logging
import json
from typing import Dict, Any, List, Optional
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
    
# 确保导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
config_dir = os.path.join(project_root, 'config')

if src_path not in sys.path:
    sys.path.append(src_path)

# 尝试导入 SerperDevTool
try:
    from crewai_tools import SerperDevTool
    serper_tool = SerperDevTool()
except ImportError:
    serper_tool = None

logger = logging.getLogger(__name__)

# 常量
DEFAULT_MODEL = "qwen-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class AgentExecutor:
    """
    Agent 执行基类。
    """
    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class CourseAgentExecutor(AgentExecutor):
    """
    使用 CrewAI 的课程生成 Agent 执行器。
    """
    def __init__(self):
        load_dotenv()
        # 加载配置
        self.agents_config = self._load_config(os.path.join(config_dir, 'course_agents.yaml'))
        self.tasks_config = self._load_config(os.path.join(config_dir, 'course_tasks.yaml'))
        self.llm = self._setup_llm()
        self.agents = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"加载配置失败 {config_path}: {e}")
        return {}

    def _setup_llm(self) -> ChatOpenAI:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
        
        # 设置 CrewAI 兼容的 OpenAI 环境变量
        os.environ["OPENAI_API_BASE"] = DEFAULT_API_BASE
        os.environ["OPENAI_BASE_URL"] = DEFAULT_API_BASE
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        return ChatOpenAI(
            model=model_name,
            base_url=DEFAULT_API_BASE,
            api_key=api_key,
            temperature=0.7
        )

    def _create_agent(self, agent_name: str) -> Agent:
        if agent_name in self.agents:
            return self.agents[agent_name]

        config = self.agents_config.get(agent_name)
        if not config:
            raise ValueError(f"未找到 Agent 配置: {agent_name}")

        tools = []
        if 'tools' in config:
            for tool_name in config['tools']:
                if tool_name == 'serper_dev_tool':
                    if serper_tool:
                        tools.append(serper_tool)
                    else:
                        logger.error(f"Agent {agent_name} 需要 serper_dev_tool 但不可用。")
                        # 根据策略，我们可能希望抛出错误或在没有工具的情况下继续
                        # 抛出错误更安全，以防止幻觉
                        raise ValueError(f"Agent {agent_name} 所需的工具 'serper_dev_tool' 不可用")

        agent = Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        self.agents[agent_name] = agent
        return agent

    def _create_task(self, task_name: str, agent: Agent, **kwargs) -> Task:
        config = self.tasks_config.get(task_name)
        if not config:
            raise ValueError(f"未找到任务配置: {task_name}")

        description = config['description'].format(**kwargs)
        expected_output = config.get('expected_output', 'Task result')

        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent
        )

    def _get_agent_output(self, result: Any) -> str:
        """
        从 CrewOutput 中提取原始文本结果。
        """
        if hasattr(result, 'raw'):
            return result.raw
        return str(result)

    def _parse_outline(self, json_str: str) -> Optional[Dict]:
        """
        解析大纲 JSON 字符串。
        """
        import re
        try:
            cleaned_str = json_str.strip()
            # 尝试查找 JSON 代码块
            json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_str, re.DOTALL)
            if json_match:
                cleaned_str = json_match.group(1)
            elif '```' in cleaned_str:
                 cleaned_str = cleaned_str.split("```")[1].strip()
            
            return json.loads(cleaned_str)
        except Exception as e:
            logger.error(f"JSON 解析失败: {e}")
            return None

    def execute(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于负载执行课程生成任务。
        负载应包含 'phase' 和阶段特定参数。
        """
        phase = task_payload.get('phase')
        
        try:
            if phase == 'research':
                return self._run_research(task_payload)
            elif phase == 'outline':
                return self._run_outline(task_payload)
            elif phase == 'chapter':
                return self._run_chapter(task_payload)
            elif phase == 'review':
                return self._run_review(task_payload)
            else:
                raise ValueError(f"未知阶段: {phase}")
        except Exception as e:
            logger.error(f"执行失败: {e}")
            raise

    def _run_research(self, payload):
        topic = payload.get('topic')
        requirements = payload.get('requirements', '')
        
        agent = self._create_agent('xiao_mei')
        task = self._create_task('research_task', agent, topic=topic, requirements=requirements)
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        output = self._get_agent_output(result)
        return {"result": output.strip()}

    def _run_outline(self, payload):
        topic = payload.get('topic')
        requirements = payload.get('requirements', '')
        chosen_direction = payload.get('chosen_direction', '')
        
        agent = self._create_agent('xiao_qing')
        task = self._create_task('outline_task', agent, 
                               chosen_direction=chosen_direction, 
                               topic=topic, 
                               requirements=requirements)
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        output = self._get_agent_output(result)
        parsed = self._parse_outline(output)
        
        if not parsed:
            # 解析失败的回退处理
            return {"result": output, "parsed": False}
            
        return {"result": parsed, "parsed": True}

    def _run_chapter(self, payload):
        chapter_index = payload.get('chapter_index')
        chapter_title = payload.get('chapter_title')
        chapter_summary = payload.get('chapter_summary')
        course_title = payload.get('course_title')
        topic = payload.get('topic')
        requirements = payload.get('requirements')
        
        agent = self._create_agent('xiao_qing')
        task = self._create_task('chapter_writing_task', agent,
                               chapter_index=chapter_index,
                               chapter_title=chapter_title,
                               chapter_summary=chapter_summary,
                               course_title=course_title,
                               topic=topic,
                               requirements=requirements)
                               
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        output = self._get_agent_output(result)
        return {"result": output}

    def _run_review(self, payload):
        course_title = payload.get('course_title')
        course_content = payload.get('course_content')
        chosen_direction = payload.get('chosen_direction')
        topic = payload.get('topic')
        requirements = payload.get('requirements')
        
        agent = self._create_agent('xiao_yin')
        task = self._create_task('review_task', agent,
                               course_title=course_title,
                               course_content=course_content,
                               chosen_direction=chosen_direction,
                               topic=topic,
                               requirements=requirements)
                               
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        output = self._get_agent_output(result)
        return {"result": output}
