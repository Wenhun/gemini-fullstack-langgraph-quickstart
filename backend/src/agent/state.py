from __future__ import annotations

from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    research_loop_count: int


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str
