import json
import re
from typing import Any, Dict, Union, List

def parse_json_from_llm(content: str) -> Union[Dict[str, Any], List[Any]]:
    """
    从 LLM 输出中解析 JSON，并处理常见问题（如 Markdown 代码块、<think> 标签与轻微格式错误）。
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
