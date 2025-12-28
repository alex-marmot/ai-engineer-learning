from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import os
from typing import List, Union
from pathlib import Path
from paddleocr import PaddleOCR
import logging

logger = logging.getLogger(__name__)

class ImageOCRReader(BaseReader):
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""
    
    def __init__(self, lang='ch', device="gpu", **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            device: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        super().__init__()
        self.lang = lang
        self.ocr = PaddleOCR(lang=lang, device=device, **kwargs)

    
    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        if isinstance(file, str):
            files = [file]
        else:
            files = file
        
        documents = []
        for image_path in files:
            try:
                if not Path(image_path).exists():
                    logger.warning(f"Image file not found: {image_path}")
                    continue

                result = self.ocr.predict(str(image_path))
                
                full_text = ""
                total_confidence = 0.0
                text_count = 0

                for res in result:
                    if isinstance(res, dict):
                        texts = res.get("rec_texts", [])
                        scores = res.get("rec_scores", [])
                        for text, score in zip(texts, scores):
                            if text:
                                full_text += text + "\n"
                                total_confidence += float(score)
                                text_count += 1

                avg_confidence = total_confidence / text_count if text_count > 0 else 0.0

                doc = Document(
                    text=full_text.strip(),
                    metadata={
                        "image_path": str(image_path),
                        "ocr_confidence_avg": avg_confidence,
                        "file_type": "image",
                        "file_name": Path(image_path).name,
                        "detected_text_count": text_count
                    }
                )
                documents.append(doc)
                logger.info(f"Successfully processed image: {image_path} (found {text_count} text blocks)")

            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                continue
        
        return documents


def initialize():
    """配置 LlamaIndex 所需的环境和模型"""
    from dotenv import load_dotenv
    from llama_index.core import Settings
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not found in .env file")

    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
        api_key=api_key
    )
    print("LlamaIndex environment setup complete.")
    
def main():
    # 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
    # 在根目录可以通过python -m ocr_research.main 运行
    data_dir = Path("data/ocr_images")
    data_dir.mkdir(parents=True, exist_ok=True)

    doc_path = data_dir / "document.png"
    ui_path = data_dir / "screenshot.png"
    scene_path = data_dir / "sign.png"
    
    image_files = [doc_path, ui_path, scene_path]

    # --- 1. 使用 ImageOCRReader 加载图像并生成 Document ---
    print("\n--- Step 1: Loading images with ImageOCRReader ---")
    reader = ImageOCRReader(lang='ch', device='cpu')  # 使用 CPU 避免 GPU 相关问题
    documents = reader.load_data(image_files)
    
    print(f"Successfully loaded {len(documents)} documents from images.")
    for doc in documents:
        print("\n--- Document ---")
        print(f"Text: {doc.text[:100]}...")
        print(f"Metadata: {doc.metadata}")

    # --- 2. 配置 LlamaIndex 环境 ---
    print("\n--- Step 2: Setting up LlamaIndex environment ---")
    initialize()

    # --- 3. 构建索引并进行查询 ---
    print("\n--- Step 3: Building index and querying ---")
    from llama_index.core import VectorStoreIndex
    
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    # 查询1: 第一封信的内容 的问题
    question1 = "第一封信的内容是什么?"
    print(f"\nQuerying: {question1}")
    response1 = query_engine.query(question1)
    print(f"Response: {response1}")

    # 查询2: 关于截图内容的问题
    question2 = "截图里灰色按钮是什么内容？"
    print(f"\nQuerying: {question2}")
    response2 = query_engine.query(question2)
    print(f"Response: {response2}")

    # 查询3: 关于路牌的问题
    question3 = "牌子上写了什么？"
    print(f"\nQuerying: {question3}")
    response3 = query_engine.query(question3)
    print(f"Response: {response3}")


if __name__ == "__main__":
    main()
