import os
import tempfile
import traceback
from pdf2image import convert_from_path
from typing import List
from app.core.llm.llm_factory import get_local_qwen_provider
from langchain_core.messages import HumanMessage

class QwenVLOCR:
    def _pdf_to_temp_images(self, file_path: str) -> List[str]:
        images = convert_from_path(file_path)
        temp_paths: List[str] = []
        for img in images:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="wb") as tmp:
                img.save(tmp.name)
                temp_paths.append(tmp.name)
        return temp_paths

    def _cleanup_files(self, paths: List[str]):
        for p in paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def process_file(self, file_path: str) -> str:
        """
        处理 PDF 或图片文件，并返回提取出的文本。
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            llm = get_local_qwen_provider()
        except Exception as e:
            print(f"获取 LLM 提供方失败：{e}")
            return ""
        
        full_text = []
        temp_files: List[str] = []

        try:
            images_paths = []
            if ext == '.pdf':
                try:
                    temp_files = self._pdf_to_temp_images(file_path)
                except Exception as e:
                    print(f"PDF 转图片失败：{e}")
                    return ""
                images_paths = temp_files
            elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
                images_paths = [file_path]
            else:
                return ""

            print(f"正在进行 OCR：共 {len(images_paths)} 张图片...")
            
            for i, img_path in enumerate(images_paths):
                abs_path = os.path.abspath(img_path).replace("\\", "/")
                image_url = f"file://{abs_path}" 

                message = HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "请准确转写这张图片中的文字。"}
                    ]
                )
                
                print(f"  - 正在处理第 {i+1}/{len(images_paths)} 张图片...")
                response = llm.invoke([message], max_new_tokens=2048)
                full_text.append(response.content)

            return "\n\n".join(full_text)

        except Exception as e:
            print(f"OCR 处理出错：{e}")
            traceback.print_exc()
            return ""
        finally:
            self._cleanup_files(temp_files)

ocr_engine = QwenVLOCR()
