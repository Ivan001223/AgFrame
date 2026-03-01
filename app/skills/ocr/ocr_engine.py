import logging
import os
import tempfile
import traceback

from langchain_core.messages import HumanMessage
from pdf2image import convert_from_path

from app.runtime.llm.llm_factory import get_local_qwen_provider

logger = logging.getLogger(__name__)


class QwenVLOCR:
    """
    基于 Qwen-VL (Vision Language) 模型的 OCR 引擎。
    支持 PDF 和常见图片格式的文字提取。
    """
    def _pdf_to_temp_images(self, file_path: str) -> list[str]:
        """
        将 PDF 转换为临时图片文件列表。
        """
        images = convert_from_path(file_path)
        temp_paths: list[str] = []
        for img in images:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="wb") as tmp:
                img.save(tmp.name)
                temp_paths.append(tmp.name)
        return temp_paths

    def _cleanup_files(self, paths: list[str]):
        """清理临时文件"""
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    logger.debug(f"Failed to remove temp file: {e}")

    def process_file(self, file_path: str) -> str:
        """
        处理 PDF 或图片文件，并返回提取出的文本。
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 提取出的文本内容
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            llm = get_local_qwen_provider()
        except Exception as e:
            logger.error(f"Failed to get LLM provider: {e}")
            return ""
        
        full_text = []
        temp_files: list[str] = []

        try:
            images_paths = []
            if ext == '.pdf':
                try:
                    temp_files = self._pdf_to_temp_images(file_path)
                except Exception as e:
                    logger.error(f"PDF to image conversion failed: {e}")
                    return ""
                images_paths = temp_files
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                images_paths = [file_path]
            else:
                return ""

            logger.info(f"Starting OCR: {len(images_paths)} images to process")
            
            for i, img_path in enumerate(images_paths):
                abs_path = os.path.abspath(img_path).replace("\\", "/")
                image_url = f"file://{abs_path}" 

                # 构造包含图片的多模态消息
                message = HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "请准确转写这张图片中的文字。"}
                    ]
                )
                
                logger.debug(f"Processing image {i+1}/{len(images_paths)}")
                response = llm.invoke([message], max_new_tokens=2048)
                full_text.append(response.content)

            return "\n\n".join(full_text)

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            traceback.print_exc()
            return ""
        finally:
            self._cleanup_files(temp_files)

ocr_engine = QwenVLOCR()
