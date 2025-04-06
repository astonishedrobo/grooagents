from langgraph.prebuilt import InjectedState
from typing import Annotated, Optional

def orchestrator(state: Annotated[dict, InjectedState], model) -> dict:
    """
    Orchestrator LLM that designs and manages the flow of the pipeline.
    """
    messages = state.get("messages", [])
    message = model.invoke(messages)
    return {'messages': [message]}