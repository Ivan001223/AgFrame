import datetime
import os

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from app.infrastructure.config.settings import settings
from app.infrastructure.utils.files import resolve_path_within_roots
from app.skills.rag.rag_engine import extract_text_from_file, get_rag_engine


# --- 1. 网页搜索 ---
@tool
def web_search(query: str) -> str:
    """
    在网络上搜索通用信息。
    """
    try:
        search = DuckDuckGoSearchRun()
        return str(search.invoke(query))
    except Exception as e:
        return f"Search failed: {e}"

# --- 2. 计算器 ---
@tool
def calculator(expression: str) -> str:
    """
    使用 Python 执行数学计算。
    输入应为合法的 Python 表达式字符串，例如 "123 * 456" 或 "math.sqrt(25)"。
    """
    try:
        flags = settings.feature_flags
        if not flags.enable_tools_python_repl:
            return "Tool disabled: calculator"
        repl = PythonREPL()
        full_code = f"import math\nprint({expression})"
        result = repl.run(full_code)
        return result.strip()
    except Exception as e:
        return f"Calculation failed: {e}"

# --- 3. 知识检索（RAG） ---
@tool
def knowledge_retriever(query: str) -> str:
    """
    从本地知识库（RAG）检索信息。
    """
    try:
        docs = get_rag_engine().retrieve_context(query)
        if not docs:
            return "No relevant information found in knowledge base."

        context_str = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in docs])
        return context_str
    except Exception as e:
        return f"Retrieval failed: {e}"

# --- 4. 文档读取（OCR） ---
@tool
def read_document(file_path: str) -> str:
    """
    读取文件并提取文本。
    优先复用统一文档解析链路，支持 PDF、图片、TXT、MD、DOCX、XLSX。
    用于分析文件内容或从图片中提取文字。

    参数:
        file_path: 文件的绝对路径。
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found {file_path}"

        text = extract_text_from_file(file_path)
        if not text:
            return "File content is empty or unreadable."
        return text
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    将文本内容写入指定文件。
    默认关闭，只有 enable_tools_write_file 打开后才会执行。
    仅允许写入受控存储目录，避免误改项目代码或系统文件。
    """
    try:
        flags = settings.feature_flags
        if not flags.enable_tools_write_file:
            return "Tool disabled: write_file"

        storage_cfg = settings.storage_local
        abs_path = resolve_path_within_roots(
            file_path,
            default_root=storage_cfg.data_dir,
            allowed_roots=(
                storage_cfg.data_dir,
                storage_cfg.documents_dir,
                storage_cfg.uploads_dir,
            ),
        )
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {abs_path}"
    except ValueError as e:
        return f"File write denied: {e}"
    except Exception as e:
        return f"File write failed: {e}"

# --- 6. 安全代码执行 ---
@tool
def python_executor(code: str) -> str:
    """
    在安全的沙箱环境中执行 Python 代码。

    参数:
        code: 要执行的 Python 代码。
    """
    try:
        flags = settings.feature_flags
        if not flags.enable_tools_python_executor:
            return "Tool disabled: python_executor"

        import asyncio

        from app.infrastructure.sandbox.code_sandbox import execute_code

        result = asyncio.run(execute_code(code))

        if result.get("success"):
            return str(result.get("output", "Code executed successfully (no output)"))
        else:
            error = result.get("error") or result.get("output", "Unknown error")
            return f"Execution failed: {error}"
    except Exception as e:
        return f"Execution error: {str(e)}"


# --- 7. 实用工具 ---
@tool
def get_current_time() -> str:
    """
    获取当前系统时间。
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


ALL_TOOLS = [
    web_search,
    calculator,
    knowledge_retriever,
    read_document,
    write_file,
    python_executor,
    get_current_time,
]
