from app.agents.node_factory import build_llm_chain, make_agent_node


def get_writer_node():
    system_prompt = """你是专业写作（Writer）Agent。
    你的目标是对对话中的内容进行润色、摘要，或进行有创意的扩写。
    
    - 若用户请求摘要：输出简洁的要点摘要。
    - 若用户请求博客文章/邮件：按合适格式排版并保证可读性。
    - 保持专业且有吸引力的语气。
    """

    chain = build_llm_chain(system_prompt, temperature=0.8)
    return make_agent_node(chain)

