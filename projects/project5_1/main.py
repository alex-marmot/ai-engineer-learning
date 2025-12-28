"""
AI 课程制作助手主入口
"""
import os
import sys
import logging
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.course_system import CourseSystem

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_environment():
    """检查环境配置"""
    # 如果有DASHSCOPE_API_KEY，也认为是有效的
    if os.getenv('DASHSCOPE_API_KEY'):
        return True

    required_vars = ['DASHSCOPE_API_KEY']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"错误：缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请在.env文件中设置这些变量")
        return False

    return True


def main():
    """主函数"""
    # 检查环境
    if not check_environment():
        return 1

    print("AI 课程制作助手")
    print("=" * 50)
    print("正在启动系统...")

    try:
        system = CourseSystem()
        system.run()
    except KeyboardInterrupt:
        print("再见!")
    except Exception as e:
        logger.error(f"系统异常: {str(e)}")
        print(f"发生错误: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
