import json
import re
from typing import Any


def parse_json_from_llm(content: str) -> dict[str, Any] | list[Any]:
    """
    从 LLM 输出中健壮地解析 JSON。
    
    能够处理：
    1. Markdown 代码块 (```json ... ```)
    2. <think> 标签（推理过程）
    3. 文本混杂
    4. 常见的格式错误（如反斜杠转义）
    
    Args:
        content: LLM 返回的原始字符串
        
    Returns:
        Union[Dict, List]: 解析后的 JSON 对象或列表
        
    Raises:
        ValueError: 如果无法解析出有效的 JSON
    """
    # 1. 移除 <think> 标签（如存在）
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # 2. 移除 Markdown 代码块
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```', '', content)
    
    # 3. 去除首尾空白
    content = content.strip()
    
    # 4. 解析 JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 若与普通文本混杂，尝试提取 JSON 对象
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 尝试修复提取 JSON 中的反斜杠转义问题
                json_str_fixed = json_str.replace('\\', '\\\\')
                try:
                    return json.loads(json_str_fixed)
                except:
                    pass
        
        # 若未找到 JSON 对象，则检查列表
        match_list = re.search(r'\[.*\]', content, re.DOTALL)
        if match_list:
            json_str = match_list.group(0)
            try:
                return json.loads(json_str)
            except:
                pass

        raise ValueError(f"Could not parse JSON from content: {content[:100]}...")
