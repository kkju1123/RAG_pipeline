from typing import TypedDict, Annotated, List, Optional
import operator

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    documents: List[str]
    current_query: str
    plan: List[str]
    language: str
