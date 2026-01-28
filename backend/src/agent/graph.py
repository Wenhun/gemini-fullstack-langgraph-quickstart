import os
from dotenv import load_dotenv
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.tools_and_schemas import SearchQueryList, Reflection
from agent.utils import get_research_topic

from agent.local_search import load_markdown_files, search_markdown


load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm(configurable: Configuration) -> ChatGroq:
    return ChatGroq(
        model=configurable.groq_model,
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
    )


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """
    Generate a set of initial search queries based on the user's question.

    This node uses a Groq‑powered language model to transform the user's message
    into a structured list of search queries. The number of queries is determined
    by the agent configuration unless explicitly overridden in the state.

    Returns:
        QueryGenerationState: A dictionary containing the generated list of search queries.
    """

    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init GROQ
    llm = get_llm(configurable)

    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState) -> List[Send]:
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    Execute a local Markdown search query and return formatted research summaries.

    This node performs one step of the research loop. For each generated search
    query, it loads (or reuses) the Markdown corpus from the configured directory
    and runs a hybrid search consisting of:
        - keyword filtering (fast, broad match)
        - semantic reranking (SentenceTransformer-based relevance scoring)

    The function returns a list of formatted summaries, where each summary contains:
        - a Markdown link to the source file
        - an excerpt of the matched content

    These summaries are later consumed by the reflection and answer-generation
    nodes in the research pipeline.

    Parameters
    ----------
    state : WebSearchState
        Contains the search query string and a unique id for this branch.
    config : RunnableConfig
        Must include `configurable["dir"]` — the path to the local Markdown
        documentation directory. The corpus is cached inside `configurable`
        to avoid reloading files on each call.

    Returns
    -------
    OverallState
        A dictionary with a single key:
            - "web_research_result": List[str]
              A list of formatted summaries ready for reflection.
              If no results are found, a placeholder summary is returned.

    Raises
    ------
    ValueError
        If the directory for local search is not provided.
    """
    configurable = config.get("configurable", {})
    base_dir = configurable.get("dir")

    if not base_dir:
        raise ValueError("No directory provided for local search (--dir).")

    if "corpus" not in configurable:
        corpus = load_markdown_files(base_dir)
        configurable["corpus"] = corpus
    else:
        corpus = configurable["corpus"]

    if raw_results := search_markdown(
        query=state["search_query"],
        corpus=corpus,
        max_results=10,
    ):
        summaries = [
            f"[{r['title']}]({r['source']})\n{r['excerpt']}" for r in raw_results
        ]
    else:
        summaries = [
            f"[no_results](no_results)\nNo results found for query: {state['search_query']}"
        ]
    return {"web_research_result": summaries}


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """
    Analyze the current research summary and identify knowledge gaps.

    This node evaluates the information gathered so far and determines whether
    additional research is needed. Using a Groq‑powered model with structured output,
    it generates follow‑up queries and flags whether the existing information is sufficient.

    Returns:
        ReflectionState: A dictionary containing sufficiency status, knowledge gaps,
        follow‑up queries, and updated loop counters.
    """

    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = get_llm(configurable)
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        Either the string "finalize_answer" or a list of Send objects
        to schedule additional web_research calls.

    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig) -> dict:
    """
    Produce the final synthesized answer based on all gathered research.

    This node composes the final response by combining the accumulated summaries
    and generating a coherent, well‑structured answer using a Groq‑powered model.
    It outputs the final AI message ready to be returned to the user.

    Returns:
        dict: A dictionary containing the final AIMessage and an empty list of sources.
    """

    configurable = Configuration.from_runnable_config(config)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model with GROQ
    llm = get_llm(configurable)
    result = llm.invoke(formatted_prompt)

    return {
        "messages": [AIMessage(content=result.content)],
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
