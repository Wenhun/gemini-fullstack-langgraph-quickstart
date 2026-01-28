import argparse
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to directory with markdown knowledge base",
    )
    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.question)],
    }

    result = graph.invoke(state, config={"configurable": {"dir": args.dir}})

    if messages := result.get("messages", []):
        print(messages[-1].content)


if __name__ == "__main__":
    main()
