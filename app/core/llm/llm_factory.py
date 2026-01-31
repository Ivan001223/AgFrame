from langchain_openai import ChatOpenAI
from app.core.config.config_manager import config_manager
from app.core.llm.local_qwen import LocalQwen2VL

# 全局单例，避免重复加载模型
_local_qwen_instance = None

def get_local_qwen_provider():
    """
    直接获取本地 Qwen 实例（单例）。
    若尚未初始化则在此完成初始化。
    
    Returns:
        LocalQwen2VL: 本地 Qwen2-VL 模型实例
    """
    global _local_qwen_instance
    if _local_qwen_instance is None:
        _local_qwen_instance = LocalQwen2VL()
    return _local_qwen_instance

def get_llm(temperature: float = 0, streaming: bool = True, json_mode: bool = False):
    """
    统一获取配置好的 LLM 实例。
    根据配置文件选择使用 OpenAI 兼容接口或本地 Qwen 模型。
    
    Args:
        temperature: 随机性参数 (0-1)
        streaming: 是否开启流式输出
        json_mode: 是否强制输出 JSON 格式
        
    Returns:
        BaseChatModel: 配置好的 LangChain 聊天模型实例
    """
    global _local_qwen_instance
    llm_config = config_manager.get_config().get("llm", {})
    model_name = llm_config.get("model", "gpt-4o")
    
    # 判断是否需要使用本地 Qwen
    if model_name == "local-qwen2-vl":
        return get_local_qwen_provider()

    model_kwargs = {}
    if json_mode and bool(llm_config.get("json_mode_response_format", True)):
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=llm_config.get("base_url"),
        api_key=llm_config.get("api_key"),
        streaming=streaming,
        model_kwargs=model_kwargs
    )
