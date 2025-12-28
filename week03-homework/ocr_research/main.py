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

                # 使用 predict() 输出的 rec_texts/rec_scores
                result = self.ocr.predict(str(image_path))
                
                full_text = ""
                total_confidence = 0
                text_count = 0

                if result:
                    for res in result:
                        data = res
                        if not isinstance(res, dict) and hasattr(res, "json"):
                            try:
                                data = res.json() if callable(res.json) else res.json
                            except Exception:
                                data = None
                        if isinstance(data, dict):
                            texts = data.get("rec_texts") or data.get("texts") or []
                            scores = data.get("rec_scores") or data.get("scores") or []
                            for text, score in zip(texts, scores):
                                if not text:
                                    continue
                                full_text += text + "\n"
                                try:
                                    total_confidence += float(score)
                                except (TypeError, ValueError):
                                    total_confidence += 0.0
                                text_count += 1
                        elif isinstance(res, (list, tuple)):
                            for line in res:
                                if line and len(line) > 1:
                                    text_info = line[1]
                                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                        text = text_info[0]
                                        confidence = text_info[1]
                                        full_text += text + "\n"
                                        total_confidence += confidence
                                        text_count += 1
                else:
                    logger.warning(f"Empty OCR result for image: {image_path}")

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
    # --- 1. 准备工作：创建示例图片 ---
    print("--- Step 1: Preparing sample images ---")
    data_dir = Path("data/ocr_images")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 简单的文本图片
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("msyh.ttc", 18) if os.path.exists("msyh.ttc") else ImageFont.load_default()
        
        # 扫描文档
        img_doc = Image.new('RGB', (600, 200), color = 'white')
        d = ImageDraw.Draw(img_doc)
        doc_text = "LlamaIndex is a powerful tool for building and querying knowledge bases. It supports multiple data sources, including text, PDF, and now even images!"
        d.text((10,10), doc_text, fill=(0,0,0), font=font)
        doc_path = data_dir / "document.png"
        img_doc.save(doc_path)

        # 屏幕截图
        img_ui = Image.new('RGB', (400, 150), color = (230, 230, 230))
        d = ImageDraw.Draw(img_ui)
        d.rectangle([10, 10, 100, 40], fill=(0, 120, 215))
        d.text((30, 15), "confirm", fill="white", font=font)
        d.text((120, 15), "username: user_test", fill="black", font=font)
        ui_path = data_dir / "screenshot.png"
        img_ui.save(ui_path)

        # 自然场景
        img_scene = Image.new('RGB', (500, 300), color = (100, 150, 200))
        d = ImageDraw.Draw(img_scene)
        d.rectangle([50, 100, 450, 200], fill='red')
        d.text((150, 130), "No Stopping", font=ImageFont.truetype("msyh.ttc", 40) if os.path.exists("msyh.ttc") else ImageFont.load_default(), fill='white')
        scene_path = data_dir / "sign.png"
        img_scene.save(scene_path)
        
        image_files = [doc_path, ui_path, scene_path]
        print(f"Sample images created in {data_dir.resolve()}")

    except (ImportError, OSError) as e:
        print(f"Pillow or font file not found, skipping image creation: {e}")
        print("Please manually place images in 'data/ocr_images' directory.")
        # 如果无法创建图片，请手动将图片放入 data/ocr_images 文件夹
        image_files = [p for p in data_dir.glob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if not image_files:
            print("No images found. Exiting.")
            return

    # --- 2. 使用 ImageOCRReader 加载图像并生成 Document ---
    print("\n--- Step 2: Loading images with ImageOCRReader ---")
    reader = ImageOCRReader(lang='en', device='cpu')  # 使用 CPU 避免 GPU 相关问题
    documents = reader.load_data(image_files)
    
    print(f"Successfully loaded {len(documents)} documents from images.")
    for doc in documents:
        print("\n--- Document ---")
        print(f"Text: {doc.text[:100]}...")
        print(f"Metadata: {doc.metadata}")

    # --- 3. 配置 LlamaIndex 环境 ---
    print("\n--- Step 3: Setting up LlamaIndex environment ---")
    initialize()

    # --- 4. 构建索引并进行查询 ---
    print("\n--- Step 4: Building index and querying ---")
    from llama_index.core import VectorStoreIndex
    
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    # 查询1: 关于 LlamaIndex 的问题
    question1 = "What is LlamaIndex?"
    print(f"\nQuerying: {question1}")
    response1 = query_engine.query(question1)
    print(f"Response: {response1}")

    # 查询2: 关于截图内容的问题
    question2 = "截图里的用户名是什么？"
    print(f"\nQuerying: {question2}")
    response2 = query_engine.query(question2)
    print(f"Response: {response2}")

    # 查询3: 关于路牌的问题
    question3 = "红色的牌子上写了什么？"
    print(f"\nQuerying: {question3}")
    response3 = query_engine.query(question3)
    print(f"Response: {response3}")


if __name__ == "__main__":
    main()
