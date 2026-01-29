from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.core.workflow.state import AgentState
from app.core.workflow.orchestrator import route_request

from app.examples.demo_researcher import get_researcher_node
from app.examples.demo_writer import get_writer_node
from app.core.tools.search_tool import get_search_tool


def run_app():
    researcher_node = get_researcher_node()
    writer_node = get_writer_node()

    tools = [get_search_tool(return_results_obj=False)]
    tool_node = ToolNode(tools)

    def orchestrator_node(state: AgentState):
        decision = route_request(state)
        print(f"编排器决策：{decision.destination}（{decision.reasoning}）")
        return {"next_step": decision.destination, "reasoning": decision.reasoning}

    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        lambda state: state["next_step"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "general": "writer",
            "FINISH": END,
        },
    )

    def should_continue_research(state):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "orchestrator"

    workflow.add_conditional_edges("researcher", should_continue_research, ["tools", "orchestrator"])
    workflow.add_edge("tools", "researcher")

    workflow.add_edge("writer", "orchestrator")

    return workflow.compile()


app = run_app()

