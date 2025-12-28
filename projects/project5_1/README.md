# AI 课程制作助手

这是一个基于 CrewAI 和 qwen-plus 大模型的智能课程制作助手。

## 功能

该系统通过三个 AI 智能体协作完成课程制作：

1.  **小美 (课程研究员)**: 负责根据用户需求搜索资料，提供课程方向建议。
2.  **小青 (课程创作者)**: 负责制定课程大纲（需用户确认）和撰写章节内容（逐章确认）。
3.  **小尹 (课程审核员)**: 负责审核最终课程内容的质量和一致性。

## 环境要求

- Python 3.10+
- 环境变量:
    - `DASHSCOPE_API_KEY`: 阿里云 DashScope API Key (用于 qwen-plus)
    - `SERPER_API_KEY`: Serper.dev API Key (用于搜索功能，可选)

## 安装

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量:
   复制 `.env.example` 为 `.env` 并填入您的 API Key。
   ```bash
   cp .env.example .env
   ```

## 使用方法

运行主程序启动助手:

```bash
python main.py
```

按照终端提示输入课程主题和要求即可。

## 项目结构

- `src/course_system.py`: 核心系统实现
- `config/course_agents.yaml`: 智能体定义 (中文)
- `config/course_tasks.yaml`: 任务定义 (中文)
- `main.py`: 程序入口
