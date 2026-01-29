from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from app.core.services.rag_engine import get_rag_engine
from app.core.services.ocr_engine import ocr_engine
import os
import datetime

# --- 1. 网页搜索 ---
@tool
def web_search(query: str) -> str:
    """
    在网络上搜索通用信息。
    """
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
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
    读取文件（PDF 或图片）并提取文本。
    用于分析文件内容或从图片中提取文字。
    
    参数:
        file_path: 文件的绝对路径。
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found {file_path}"
            
        text = ocr_engine.process_file(file_path)
        if not text:
            return "File content is empty or unreadable."
        return text
    except Exception as e:
        return f"Error reading file: {str(e)}"

# --- 5. 文件系统工具 ---
@tool
def write_file(file_path: str, content: str) -> str:
    """
    将文本内容写入文件；若文件已存在则覆盖。
    
    参数:
        file_path: 文件的绝对路径。
        content: 要写入的文本内容。
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to file: {file_path}"
    except Exception as e:
        return f"Write failed: {str(e)}"

# --- 6. 实用工具 ---
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
    get_current_time
]
