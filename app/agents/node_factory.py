from typing import Any, Callable, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.llm.llm_factory import get_llm
from app.core.workflow.state import AgentState


def build_system_prompt_template(system_prompt: str, messages_key: str = "messages") -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name=messages_key),
        ]
    )


def build_llm_chain(
    system_prompt: str,
    *,
    temperature: float = 0,
    tools: Optional[Sequence[Any]] = None,
    json_mode: bool = False,
):
    llm = get_llm(temperature=temperature, json_mode=json_mode)
    if tools:
        llm = llm.bind_tools(list(tools))
    prompt = build_system_prompt_template(system_prompt)
    return prompt | llm


def make_agent_node(chain, *, messages_key: str = "messages") -> Callable[[AgentState], dict]:
    def node(state: AgentState):
        messages = state[messages_key]
        response = chain.invoke({messages_key: messages})
        return {"messages": [response]}

    return node

