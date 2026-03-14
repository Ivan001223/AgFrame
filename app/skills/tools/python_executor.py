from typing import Any

from app.skills.common.tools import register_tool


@register_tool("python_executor")
async def python_executor(code: str) -> dict[str, Any]:
    """
    Execute Python code in a secure sandbox environment.

    Args:
        code: Python code to execute

    Returns:
        Dict containing:
        - success: bool indicating if execution succeeded
        - output: stdout from the code
        - error: error message if any
    """
    from app.infrastructure.sandbox.code_sandbox import execute_code

    result = await execute_code(code)

    if result.get("success"):
        return {
            "success": True,
            "output": result.get("output", ""),
            "type": "text",
        }
    else:
        return {
            "success": False,
            "error": result.get("error") or result.get("output", "Unknown error"),
            "type": "text",
        }
