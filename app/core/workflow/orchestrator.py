from typing import Dict, Any
from pydantic import BaseModel, Field

from app.core.workflow.json_router import run_json_router

class RouteDecision(BaseModel):
    """通用路由决策模型"""
    destination: str = Field(description="The next node/agent to route to (e.g., 'agent_a', 'agent_b', 'FINISH')")
    reasoning: str = Field(description="Reasoning for the routing decision")

def route_request(state: Dict[str, Any]) -> RouteDecision:
    """
    通用编排器/路由节点。
    分析对话历史以决定下一步动作。
    """
    messages = state.get("messages", [])
    if not messages:
        return RouteDecision(destination="general", reasoning="No messages found")

    # 在此定义各个 Agent 的能力
    # 从一个通用系统提示词开始
    # 提示词框架示例（Demo：路由/编排器）
    # - 角色：监督者/路由器
    # - 目标：根据输入判定 next_step（以及可选 reasoning）
    # - 输入：对话消息（必要时先做清洗/裁剪以提高稳定性）
    # - 规则：给出明确的条件分支（何时 FINISH、何时分流到具体 Agent）
    # - 输出：严格的机器可解析格式（例如仅输出 JSON，字段名固定）
    # - 约束：禁止输出额外文本、禁止 Markdown 代码块、禁止解释过程
    system_template = """你是监督者（Supervisor）Agent。
    你的目标是将用户请求路由到合适的子 Agent；当任务已完成时返回 FINISH。

    ### 可用 Agent：
    1. **general**：处理一般聊天与不需要特定工具的请求。

    ### 路由规则：
    - 若用户意图清晰且匹配某个 Agent，则路由到该 Agent。
    - 若请求已完成或对话应结束，则返回 \"FINISH\"。

    ### 输出要求：
    - 仅输出符合 schema 的合法 JSON（不要输出额外文本、Markdown 代码块或解释）。
    """
    return run_json_router(
        messages,
        system_template=system_template,
        schema=RouteDecision,
        fallback_data={"destination": "general", "reasoning": "Error: {error}"},
        temperature=0,
        streaming=False,
        json_mode=True,
    )
