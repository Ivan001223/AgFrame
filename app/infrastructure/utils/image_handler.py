import base64
import mimetypes
from typing import Any

import requests


def is_local_url(url: str) -> bool:
    """检查 URL 是否指向本地服务器。"""
    return url.startswith('http://localhost') or url.startswith('http://127.0.0.1')

def convert_url_to_base64(url: str) -> str:
    """
    从 URL 拉取图片并转换为 base64 data URI。
    适用于将本地图片传给云端 LLM。
    """
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            mime_type = mimetypes.guess_type(url)[0] or 'image/jpeg'
            b64_data = base64.b64encode(resp.content).decode('utf-8')
            return f"data:{mime_type};base64,{b64_data}"
    except Exception as e:
        print(f"将图片转换为 base64 失败：{e}")
    return None

def process_multimodal_content(raw_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    处理多模态消息内容列表；必要时将本地图片 URL 转为 base64 data URI。
    """
    processed_content = []
    for item in raw_content:
        if item.get('type') == 'image_url':
            url = item['image_url']['url']
            if is_local_url(url):
                data_uri = convert_url_to_base64(url)
                if data_uri:
                    processed_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                else:
                    # 兜底：保留原始内容（可能失败）
                    processed_content.append(item)
            else:
                processed_content.append(item)
        else:
            processed_content.append(item)
    return processed_content
