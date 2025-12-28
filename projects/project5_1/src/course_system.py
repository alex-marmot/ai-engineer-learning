import os
import sys
import yaml
import json
import logging
import re
from typing import Dict, List, Any, Optional
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_MODEL = "qwen-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHECKPOINT_FILE = "course_checkpoint.json"
CONFIG_DIR = "config"

class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """
    自定义回调处理程序，用于将 LLM 的输出流式传输到标准输出。
    """
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

# 尝试导入工具，如果不可用则记录警告
# SerperDevTool 用于搜索互联网
try:
    from crewai_tools import SerperDevTool
    serper_tool = SerperDevTool()
except ImportError:
    logger.warning("SerperDevTool not available (未安装 crewai_tools 或配置错误)")
    serper_tool = None

class CourseSystem:
    """
    课程生成系统主类。
    负责管理 Agent、Task 以及整个课程生成的流程（研究、大纲、章节编写、审核）。
    """

    def __init__(self):
        """初始化课程系统，加载配置和 LLM。"""
        # 加载 Agent 和 Task 的配置文件
        self.agents_config = self._load_config(f'{CONFIG_DIR}/course_agents.yaml')
        self.tasks_config = self._load_config(f'{CONFIG_DIR}/course_tasks.yaml')
        
        # 初始化 LLM
        self.llm = self._setup_llm()
        
        # 缓存已创建的 Agent
        self.agents: Dict[str, Agent] = {}
        
        # 检查点文件路径
        self.checkpoint_file = CHECKPOINT_FILE

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        从 YAML 文件加载配置。
        
        Args:
            config_path: 配置文件路径 (相对或绝对)
            
        Returns:
            配置字典，如果加载失败则返回空字典
        """
        # 尝试多个可能的路径寻找配置文件
        possible_paths = [
            config_path,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path), # 相对于 src 的上一级
            os.path.join(os.getcwd(), config_path), # 相对于当前工作目录
            os.path.join(os.path.dirname(__file__), "..", config_path) # 另一种相对路径
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        logger.info(f"成功加载配置文件: {path}")
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.error(f"读取配置文件 {path} 失败: {e}")
                    return {}
        
        logger.error(f"未找到配置文件: {config_path}，已尝试路径: {possible_paths}")
        return {}

    def _setup_llm(self) -> ChatOpenAI:
        """
        配置并初始化 LLM (使用 DashScope/通义千问)。
        
        Returns:
            配置好的 ChatOpenAI 实例
        """
        api_key = os.environ.get("DASHSCOPE_API_KEY")

        if not api_key:
            logger.warning("未在环境变量中找到 DASHSCOPE_API_KEY，请确保已设置。")
        else:
            # 掩码显示 API Key，用于日志
            masked_key = f"{api_key[:8]}******{api_key[-4:]}" if len(api_key) > 12 else "******"
            logger.info(f"已加载 DASHSCOPE_API_KEY: {masked_key}")
        
        # 设置 OpenAI 兼容的环境变量
        os.environ["OPENAI_API_BASE"] = DEFAULT_API_BASE
        os.environ["OPENAI_BASE_URL"] = DEFAULT_API_BASE
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # 获取模型名称，默认为 qwen-plus
        model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)

        return ChatOpenAI(
            model=model_name,
            base_url=DEFAULT_API_BASE,
            api_key=api_key,
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

    def _create_agent(self, agent_name: str) -> Agent:
        """
        根据配置创建 CrewAI Agent。
        
        Args:
            agent_name: agent 名称 (需在 config 中存在)
            
        Returns:
            创建的 Agent 实例
        """
        if agent_name in self.agents:
            return self.agents[agent_name]

        config = self.agents_config.get(agent_name)
        if not config:
            raise ValueError(f"未找到 Agent 配置: '{agent_name}'")

        # 配置工具
        tools = []
        if 'tools' in config:
            for tool_name in config['tools']:
                if tool_name == 'serper_dev_tool' and serper_tool:
                    tools.append(serper_tool)

        agent = Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            mcps=[
                "https://mcp.exa.ai/mcp?api_key=6b39017d-ff47-45bb-9c61-a54f3011da54",           # External MCP server
                "https://api.weather.com/mcp#get_forecast",          # Specific tool from server
                "crewai-amp:financial-data",                         # CrewAI AOP marketplace 财务能力
                "crewai-amp:research-tools#pubmed_search"            # Specific AMP tool 搜索生物医学文献的能力
            ]
        )
        self.agents[agent_name] = agent
        return agent

    def _create_task(self, task_name: str, agent: Agent, **kwargs) -> Task:
        """
        根据配置创建 CrewAI Task。
        
        Args:
            task_name: task 名称 (需在 config 中存在)
            agent: 指派的 Agent
            **kwargs: 用于格式化任务描述的参数
            
        Returns:
            创建的 Task 实例
        """
        config = self.tasks_config.get(task_name)
        if not config:
            raise ValueError(f"未找到 Task 配置: '{task_name}'")

        # 格式化任务描述
        try:
            description = config['description'].format(**kwargs)
        except KeyError as e:
            logger.error(f"格式化任务 '{task_name}' 描述失败，缺少参数: {e}")
            description = config['description'] # 回退到原始描述

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
        解析大纲生成的 JSON 字符串。
        增强了对 Markdown 代码块的处理。
        """
        try:
            # 清理 Markdown 代码块标记
            cleaned_str = json_str.strip()
            
            # 尝试使用正则提取 JSON 部分
            json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_str, re.DOTALL)
            if json_match:
                cleaned_str = json_match.group(1)
            elif '```' in cleaned_str:
                 # 处理没有 json 标签的代码块
                 cleaned_str = cleaned_str.split("```")[1].strip()
            
            return json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n原始内容: {json_str[:200]}...")
            return None
        except Exception as e:
            logger.error(f"解析大纲时发生未知错误: {e}")
            return None

    def _save_checkpoint(self, state: Dict):
        """保存当前会话状态到文件。"""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            # logger.info("检查点已保存") # 减少日志噪音
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def _load_checkpoint(self) -> Optional[Dict]:
        """从文件加载会话状态。"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
        return None

    def _clear_checkpoint(self):
        """清除会话状态文件。"""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("检查点已清除")
            except Exception as e:
                logger.error(f"清除检查点失败: {e}")

    def _run_research_phase(self, topic: str, requirements: str) -> Optional[str]:
        """
        阶段 1: 市场调研
        让 Agent 'xiao_mei' 搜索并建议课程方向。
        """
        print("\n🔍 小美正在搜索课程方向...")
        xiao_mei = self._create_agent('xiao_mei')
        research_task = self._create_task('research_task', xiao_mei, topic=topic, requirements=requirements)
        
        crew = Crew(
            agents=[xiao_mei],
            tasks=[research_task],
            verbose=True,
            process=Process.sequential
        )
        try:
            # 执行研究任务
            research_result = crew.kickoff()
            output = self._get_agent_output(research_result)
            print("\n📋 建议的课程方向:")
            print(output)
            
            print("\n-------------------------------------------")
            print("请从上方选择一个方向。")
            return input("请输入您选择的方向 (复制粘贴或描述): ").strip()
        except Exception as e:
            print(f"❌ 研究阶段出错: {e}")
            logger.exception("Research phase error")
            return None

    def _run_outline_phase(self, topic: str, requirements: str, chosen_direction: str) -> Optional[Dict]:
        """
        阶段 2: 大纲制定
        让 Agent 'xiao_qing' 根据选定方向生成课程大纲。
        """
        xiao_qing = self._create_agent('xiao_qing')
        current_requirements = requirements
        
        while True:
            # 执行大纲任务
            print("\n📝 小青正在制定课程大纲...")
            outline_task = self._create_task(
                'outline_task', 
                xiao_qing, 
                chosen_direction=chosen_direction, 
                topic=topic, 
                requirements=current_requirements
            )
            
            crew = Crew(
                agents=[xiao_qing],
                tasks=[outline_task],
                verbose=True,
                process=Process.sequential
            )
            try:
                outline_result = crew.kickoff()
                outline_str = self._get_agent_output(outline_result)
                outline_data = self._parse_outline(outline_str)
                
                if not outline_data or not outline_data.get('chapters'):
                    print("\n⚠️ 大纲解析失败或格式不正确。")
                    retry = input("是否重试? (y/n): ").strip().lower()
                    if retry == 'y':
                        continue
                    return None

                print("\n📋 生成的课程大纲:")
                chapters = outline_data.get('chapters', [])
                for i, chapter in enumerate(chapters):
                    print(f"第 {i+1} 章: {chapter['title']} - {chapter['summary']}")

                while True:
                    choice = input("\n您可以: [1] 确认继续 [2] 修改要求重生成 [3] 退出 (请输入数字): ").strip()
                    if choice == '1':
                        # 确保课程标题存在
                        if 'course_title' not in outline_data:
                            outline_data['course_title'] = chosen_direction
                        return outline_data
                    elif choice == '2':
                        feedback = input("请输入修改建议: ").strip()
                        current_requirements = f"{requirements}\n修改建议: {feedback}"
                        break # 跳出内层循环，重新生成
                    elif choice == '3':
                        return None
                    else:
                        print("无效输入，请重试。")
            except Exception as e:
                logger.error(f"Outline generation failed: {e}")
                print(f"❌ 生成出错: {e}")
                retry = input("是否重试? (y/n): ").strip().lower()
                if retry != 'y':
                    return None

    def _run_chapter_phase(self, outline_data: Dict, topic: str, requirements: str, state: Dict) -> Optional[List[str]]:
        """
        阶段 3: 章节内容生成
        逐章生成内容，并允许用户审核/修改。
        """
        chapters = outline_data.get('chapters', [])
        course_title = outline_data.get('course_title', "未命名课程")
        
        # 初始化内容列表（如果不存在或长度不匹配）
        course_content = state.get('course_content')
        if not course_content or len(course_content) != len(chapters):
            course_content = [None] * len(chapters)
            state['course_content'] = course_content
        
        xiao_qing = self._create_agent('xiao_qing')
        
        i = 0
        while i < len(chapters):
            # 如果该章节已有内容，跳过
            if course_content[i] is not None:
                i += 1
                continue

            chapter = chapters[i]
            print(f"\n✍️ 正在生成第 {i+1} 章: {chapter['title']}...")
            
            chapter_requirements = requirements
            while True:
                chapter_task = self._create_task(
                    'chapter_writing_task', 
                    xiao_qing, 
                    chapter_index=i+1,
                    chapter_title=chapter['title'],
                    chapter_summary=chapter['summary'],
                    course_title=course_title,
                    topic=topic,
                    requirements=chapter_requirements
                )
                
                crew = Crew(
                    agents=[xiao_qing],
                    tasks=[chapter_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                try:
                    chapter_result = crew.kickoff()
                    content_str = self._get_agent_output(chapter_result)
                    
                    print(f"\n📄 第 {i+1} 章内容预览:")
                    print(content_str[:500] + "...\n(内容已截断)")
                    
                    choice = input(f"\n针对第 {i+1} 章，您可以: [1] 确认 [2] 修改/重写 [3] 退出 (请输入数字): ").strip()
                    if choice == '1':
                        course_content[i] = f"# 第 {i+1} 章: {chapter['title']}\n\n{content_str}"
                        # 确认后保存检查点
                        state['course_content'] = course_content
                        self._save_checkpoint(state)
                        break 
                    elif choice == '2':
                        feedback = input("请输入修改建议: ").strip()
                        chapter_requirements = f"{requirements}\n针对本章的修改建议: {feedback}"
                        continue # 重新生成当前章节
                    elif choice == '3':
                        return None
                    else:
                        print("无效输入")
                except Exception as e:
                    print(f"❌ 生成出错: {e}")
                    logger.exception("Chapter generation error")
                    retry = input("是否重试本章? (y/n): ").strip().lower()
                    if retry != 'y':
                        return None
            
            i += 1
            
        return course_content

    def _run_review_phase(self, course_content: List[str], course_title: str, chosen_direction: str, topic: str, requirements: str):
        """
        阶段 4: 全文审核
        让 Agent 'xiao_yin' 审核整个课程内容。
        """
        # 过滤掉 None 值，防止 course_content 列表中存在空章节导致拼接失败
        valid_content = [c for c in course_content if c]
        full_content = "\n\n".join(valid_content)
        
        print("\n🧐 小尹正在审核课程...")
        xiao_yin = self._create_agent('xiao_yin')
        review_task = self._create_task(
            'review_task', 
            xiao_yin, 
            course_title=course_title, 
            course_content=full_content,
            chosen_direction=chosen_direction,
            topic=topic,
            requirements=requirements
        )
        
        crew = Crew(
            agents=[xiao_yin],
            tasks=[review_task],
            verbose=True,
            process=Process.sequential
        )
        
        try:
            review_result = crew.kickoff()
            report = self._get_agent_output(review_result)
            
            print("\n最终审核报告:")
            print(report)
            
            return report, full_content
        except Exception as e:
            print(f"❌ 审核出错: {e}")
            logger.exception("Review phase error")
            return None, full_content

    def run(self):
        """
        运行课程制作工作流的主入口。
        """
        print("===========================================")
        print("       欢迎使用 AI 课程制作助手！")
        print("===========================================")
        
        # 1. 加载或初始化状态
        state = self._load_checkpoint()
        if state:
            resume = input("检测到未完成的会话。是否恢复? (y/n): ").strip().lower()
            if resume == 'y':
                topic = state.get('topic')
                requirements = state.get('requirements')
                chosen_direction = state.get('chosen_direction')
                outline_data = state.get('outline_data')
                print(f"已恢复主题: {topic}")
            else:
                state = {}
                self._clear_checkpoint()
        else:
            state = {}

        # 获取基础信息
        if not state.get('topic'):
            topic = input("请输入课程主题: ").strip()
            requirements = input("请输入具体要求: ").strip()
            state['topic'] = topic
            state['requirements'] = requirements
            self._save_checkpoint(state)
        else:
            topic = state['topic']
            requirements = state['requirements']
        
        # 2. 研究阶段 (Research)
        if not state.get('chosen_direction'):
            chosen_direction = self._run_research_phase(topic, requirements)
            if not chosen_direction: return
            state['chosen_direction'] = chosen_direction
            self._save_checkpoint(state)
        else:
            chosen_direction = state['chosen_direction']
            print(f"已恢复方向: {chosen_direction}")

        # 3. 大纲阶段 (Outline)
        if not state.get('outline_data'):
            outline_data = self._run_outline_phase(topic, requirements, chosen_direction)
            if not outline_data: return
            state['outline_data'] = outline_data
            self._save_checkpoint(state)
        else:
            outline_data = state['outline_data']
            print("已恢复大纲。")

        # 4. 章节生成阶段 (Chapter Generation)
        course_content = self._run_chapter_phase(outline_data, topic, requirements, state)
        if not course_content:
            return

        # 5. 审核与保存循环 (Review & Save)
        while True:
            report, full_content = self._run_review_phase(
                course_content, 
                outline_data.get('course_title', 'Course'), 
                chosen_direction, 
                topic, 
                requirements
            )
            
            if not report:
                print("审核失败。")
                break

            print("\n-------------------------------------------")
            choice = input("根据审核报告，您可以: [1] 通过并保存 [2] 修改特定章节 [3] 退出 (请输入数字): ").strip()
            
            if choice == '1':
                # 保存文件
                safe_title = re.sub(r'[\\/*?:"<>|]', "", outline_data.get('course_title', 'Course')).strip()
                filename = f"{safe_title}.txt"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(full_content)
                    print(f"\n💾 课程内容已保存至: {filename}")
                    print("\n🎉 课程制作流程完成！")
                    self._clear_checkpoint()
                except Exception as e:
                    print(f"❌ 保存文件失败: {e}")
                return
                
            elif choice == '2':
                # 修改特定章节
                try:
                    chapter_idx = int(input(f"请输入要修改的章节序号 (1-{len(course_content)}): ").strip()) - 1
                    if 0 <= chapter_idx < len(course_content):
                        # 清除该章节内容以强制重新生成
                        course_content[chapter_idx] = None
                        print(f"已标记第 {chapter_idx+1} 章为待修改。")
                        
                        # 更新状态
                        state['course_content'] = course_content
                        self._save_checkpoint(state)
                        
                        # 重新进入章节生成阶段 (会自动跳过已存在的章节)
                        course_content = self._run_chapter_phase(outline_data, topic, requirements, state)
                        if not course_content:
                            return
                    else:
                        print("无效的章节序号。")
                except ValueError:
                    print("请输入有效的数字。")
            elif choice == '3':
                return
            else:
                print("无效输入。")

if __name__ == "__main__":
    try:
        system = CourseSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\n程序已由用户中断。")
    except Exception as e:
        logger.exception("程序发生未捕获的异常")
        print(f"\n❌ 程序发生错误: {e}")
