# OCR 实验报告

## 1. 架构设计图

### 1.1 ImageOCRReader 在 LlamaIndex 流程中的位置

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LlamaIndex 数据处理流程                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │  数据源      │───▶│  ImageOCRReader │───▶│   Document       │   │
│  │  (图像文件)  │    │   (OCR 读取器)   │    │  (标准化文档)     │   │
│  └─────────────┘    └─────────────────┘    └──────────────────┘   │
│         │                    │                    │                │
│         │                    │                    │                │
│         │                    │                    ▼                │
│         │                    │         ┌──────────────────┐          │
│         │                    │         │  VectorStoreIndex│          │
│         │                    │         │    (向量索引)     │          │
│         │                    │         └──────────────────┘          │
│         │                    │                    │                │
│         │                    │                    ▼                │
│         │                    │         ┌──────────────────┐          │
│         │                    │         │  Query Engine    │          │
│         │                    │         │   (查询引擎)      │          │
│         │                    │         └──────────────────┘          │
│         │                    │                    │                │
│         │                    │                    ▼                │
│         │                    │         ┌──────────────────┐          │
│         │                    │         │   查询结果        │          │
│         │                    │         │  (文本回答)       │          │
│         │                    │         └──────────────────┘          │
│         │                    │                                     │
│         │                    │                                     │
│         │                    │   ┌──────────────────────────┐      │
│         │                    │   │  PaddleOCR               │      │
│         │                    └──▶│  (PP-OCR v5 引擎)         │      │
│         │                        └──────────────────────────┘      │
│         │                                                           │
│         └───────────────────────────────────────────────────────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 架构说明

**ImageOCRReader** 在 LlamaIndex 生态系统中扮演着**自定义数据读取器**的角色：

1. **继承体系**：继承自 `llama_index.core.readers.base.BaseReader`，遵循 LlamaIndex 的标准读取器接口规范
2. **数据转换**：将非结构化图像数据转换为结构化 Document 对象，实现图像文本内容的向量化检索
3. **位置作用**：位于数据摄入管道的前端，负责将图像文件解析为 LlamaIndex 可处理的文本格式

**核心处理流程**：
- 接收图像文件路径（支持批量处理）
- 调用 PaddleOCR 引擎进行文本识别
- 提取识别结果中的文本内容和置信度分数
- 封装为带元数据的 Document 对象
- 输出标准化文档供后续索引构建使用

## 2. 核心代码说明

### ImageOCRReader 类代码逐行注释

```python
from llama_index.core.readers.base import BaseReader  # 导入 LlamaIndex 基础读取器类
from llama_index.core.schema import Document          # 导入 LlamaIndex 文档对象类
import os                                             # 导入操作系统接口
from typing import List, Union                        # 导入类型提示工具
from pathlib import Path                              # 导入路径操作工具
from paddleocr import PaddleOCR                       # 导入 PaddleOCR 引擎
import logging                                        # 导入日志记录模块

logger = logging.getLogger(__name__)                  # 创建当前模块的日志记录器

class ImageOCRReader(BaseReader):                     # 定义图像 OCR 读取器类，继承自 BaseReader
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""  # 类文档字符串
    
    def __init__(self, lang='ch', device="gpu", **kwargs):  # 构造函数，初始化 OCR 引擎
        """
        Args:                                         # 参数说明
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)   # 指定识别语言
            device: 是否使用 GPU 加速                   # 指定设备类型
            **kwargs: 其他传递给 PaddleOCR 的参数      # 扩展参数
        """
        super().__init__()                           # 调用父类构造函数
        self.lang = lang                             # 保存语言设置
        self.ocr = PaddleOCR(lang=lang, device=device, **kwargs)  # 初始化 PaddleOCR 引擎

    
    def load_data(self, file: Union[str, List[str]]) -> List[Document]:  # 核心数据处理函数
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:                                        # 参数说明
            file: 图像路径字符串 或 路径列表        # 支持单文件和多文件
        Returns:                                     # 返回值
            List[Document]                          # 文档对象列表
        """
        if isinstance(file, str):                   # 判断输入是否为字符串（单文件）
            files = [file]                          # 转换为单元素列表
        else:
            files = file                            # 直接使用文件列表
        
        documents = []                              # 初始化文档列表
        for image_path in files:                    # 遍历所有图像文件
            try:                                    # 异常捕获，确保单个文件失败不影响其他
                if not Path(image_path).exists():   # 检查文件是否存在
                    logger.warning(f"Image file not found: {image_path}")  # 记录警告日志
                    continue                        # 跳过不存在的文件

                # PaddleOCR 3.3.2 predict() 返回生成器，每个元素是 dict
                result = self.ocr.predict(str(image_path))  # 调用 OCR 识别
                
                full_text = ""                    # 初始化完整文本
                total_confidence = 0.0            # 初始化总置信度
                text_count = 0                    # 初始化文本块计数

                for res in result:                # 遍历 OCR 结果生成器
                    if isinstance(res, dict):     # 检查结果是否为字典
                        texts = res.get("rec_texts", [])      # 获取识别文本列表
                        scores = res.get("rec_scores", [])    # 获取置信度分数列表
                        for text, score in zip(texts, scores):  # 并行遍历文本和分数
                            if text:              # 检查文本是否非空
                                full_text += text + "\n"  # 累加文本，每块换行
                                total_confidence += float(score)  # 累加置信度
                                text_count += 1   # 计数器加一

                avg_confidence = total_confidence / text_count if text_count > 0 else 0.0  # 计算平均置信度

                doc = Document(                   # 创建文档对象
                    text=full_text.strip(),       # 设置文档文本（去除首尾空白）
                    metadata={                    # 设置元数据字典
                        "image_path": str(image_path),        # 图像文件路径
                        "ocr_confidence_avg": avg_confidence,  # 平均置信度
                        "file_type": "image",                 # 文件类型
                        "file_name": Path(image_path).name,   # 文件名
                        "detected_text_count": text_count     # 检测到的文本块数量
                    }
                )
                documents.append(doc)             # 将文档添加到列表
                logger.info(f"Successfully processed image: {image_path} (found {text_count} text blocks)")  # 记录成功日志

            except Exception as e:              # 捕获所有异常
                logger.error(f"Failed to process image {image_path}: {e}")  # 记录错误日志
                continue                        # 继续处理下一个文件
        
        return documents                        # 返回文档列表
```

## 3. OCR 效果评估

Querying: 第一封信的内容是什么?
Response: 第一封信的内容是：“他欺骗了她。”

Querying: 截图里灰色按钮是什么内容？
Response: 取消

Querying: 牌子上写了什么？
Response: 终点

## 4.错误案例分析

最后那张图只识别出了终点两个字，其他小一些文字没有识别出来不知道这是不是算复杂背景。

## 5. Document 封装合理性讨论：文本拼接方式是否合理？元数据设计是否有助于后续检索？

### 5.1 文本拼接方式

当前实现将 OCR 识别得到的文本块按返回顺序拼接，并使用换行符分隔，最终封装为单一 Document。这种方式实现简单、稳定，能够保证 OCR 文本完整进入索引，适合作为工程上的初始方案。
在扫描文档或线性排版内容中，该拼接方式基本可用。但在 UI 截图、表格或多列排版场景下，OCR 返回顺序不一定符合阅读顺序，简单拼接可能导致语义顺序错乱。此外，统一使用换行符难以区分段落、字段或按钮等不同语义单元，容易引入冗余信息。
总体来看，该拼接方式在工程上可行，但对复杂版面支持有限，更适合作为基线实现。

### 5.2 元数据设计

当前元数据包含图像路径、文件名、OCR 平均置信度及识别文本块数量，能够支持结果回溯与文本质量评估，对工程调试和实验复现有帮助。
但这些元数据主要用于记录信息，对检索阶段的直接辅助有限。缺少文本块位置、区域类型等结构信息，使系统难以在后续阶段利用页面布局进行更精确的上下文定位。

## 6. 局限性与改进建议：如何保留空间结构（如表格）？是否可加入 layout analysis（如 PP-Structure）？

当前实现的主要局限在于未保留页面的空间结构信息。二维版面被压缩为一维文本后，表格行列关系、界面控件位置等关键信息无法体现，容易影响问答的准确性。

针对这一问题，可在 OCR 后引入布局分析能力（如 PP-Structure），对页面进行区域划分，并对不同区域采用不同处理方式：普通文本按阅读顺序拼接，表格区域保留结构并输出为 Markdown 或 CSV。这样可以在不改变整体流程的前提下，显著降低无关上下文对检索和生成的干扰。

在工程演进上，还可结合 OCR 置信度对低质量文本进行过滤或降权，并逐步引入块级或区域级索引，以提升复杂页面场景下的稳定性。