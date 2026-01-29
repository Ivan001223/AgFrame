from app.core.tools.search_tool import get_search_tool
from app.agents.node_factory import build_llm_chain, make_agent_node


def get_researcher_node():
    tools = [get_search_tool(return_results_obj=False)]

    system_prompt = """你是研究员（Researcher）Agent。
    你的目标是通过检索与综合信息，回答用户的问题，并尽量保持信息准确与及时。

    - 当需要外部信息时，务必使用搜索工具（例如 duckduckgo_results_json 或类似工具）。
    - 清晰总结搜索结果，并给出可执行/可验证的结论。
    - 当信息已经足够时，直接给出完整回答；当信息不足时，明确说明缺口与下一步建议。
    """

    chain = build_llm_chain(system_prompt, temperature=0.7, tools=tools)
    return make_agent_node(chain)

