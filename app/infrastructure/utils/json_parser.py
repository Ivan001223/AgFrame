import json
import re
from typing import Any


def _return_json_value(value: Any) -> dict[str, Any] | list[Any] | None:
    if isinstance(value, (dict, list)):
        return value
    return None


def parse_json_from_llm(content: str) -> dict[str, Any] | list[Any]:
    """
    Parse JSON robustly from LLM output.

    Handles markdown fences, <think> blocks, mixed prose, and common bad
    backslash escaping in extracted JSON snippets.
    """
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```", "", content).strip()

    last_error: Exception | None = None
    try:
        parsed = _return_json_value(json.loads(content))
        if parsed is not None:
            return parsed
    except json.JSONDecodeError as exc:
        last_error = exc

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            parsed = _return_json_value(json.loads(json_str))
            if parsed is not None:
                return parsed
        except json.JSONDecodeError as exc:
            last_error = exc

        json_str_fixed = json_str.replace("\\", "\\\\")
        try:
            parsed = _return_json_value(json.loads(json_str_fixed))
            if parsed is not None:
                return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    match_list = re.search(r"\[.*\]", content, re.DOTALL)
    if match_list:
        json_str = match_list.group(0)
        try:
            parsed = _return_json_value(json.loads(json_str))
            if parsed is not None:
                return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    raise ValueError(f"Could not parse JSON from content: {content[:100]}...") from last_error
