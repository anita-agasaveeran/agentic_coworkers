"""DailyTechUpgradeAgent: LangGraph wiring and command-line entry point.

    python agent.py                     # today's brief, asks for ratings at the end
    python agent.py --date 2026-10-29   # re-run a specific day
    python agent.py --no-feedback       # skip the rating step
"""

import argparse
from datetime import date

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import nodes
from config import CHECKPOINT_DB, DATA_DIR
from state import BriefState


def build_graph(checkpointer=None):
    graph = StateGraph(BriefState)

    graph.add_node("load_profile", nodes.load_profile)
    graph.add_node("plan_queries", nodes.plan_queries)
    graph.add_node("fetch_candidates", nodes.fetch_candidates)
    graph.add_node("dedupe_and_filter", nodes.dedupe_and_filter)
    graph.add_node("score_and_rank", nodes.score_and_rank)
    graph.add_node("broaden_queries", nodes.broaden_queries)
    graph.add_node("extract_full_text", nodes.extract_full_text)
    graph.add_node("summarize_and_personalize", nodes.summarize_and_personalize)
    graph.add_node("compose_brief", nodes.compose_brief)
    graph.add_node("collect_feedback", nodes.collect_feedback)
    graph.add_node("update_memory", nodes.update_memory)

    graph.add_edge(START, "load_profile")
    graph.add_edge("load_profile", "plan_queries")
    graph.add_edge("plan_queries", "fetch_candidates")
    graph.add_edge("fetch_candidates", "dedupe_and_filter")
    graph.add_edge("dedupe_and_filter", "score_and_rank")
    graph.add_conditional_edges("score_and_rank", nodes.route_after_scoring)
    graph.add_edge("broaden_queries", "fetch_candidates")
    graph.add_edge("extract_full_text", "summarize_and_personalize")
    graph.add_edge("summarize_and_personalize", "compose_brief")
    graph.add_edge("compose_brief", "collect_feedback")
    graph.add_edge("collect_feedback", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile(checkpointer=checkpointer)


def ask_for_feedback(payload: dict) -> list[dict]:
    """Terminal prompt for the collect_feedback interrupt."""
    print(f"\nBrief saved to {payload.get('brief_path')}")
    feedback = []
    for i, article in enumerate(payload.get("articles", []), 1):
        raw = input(f"Rate {i}. {article['title']} (1-5, Enter to skip): ").strip()
        if not raw:
            continue
        tags = input("  Tags (comma-separated, e.g. more, less, too basic): ").strip()
        feedback.append({
            "url": article["url"],
            "rating": int(raw),
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
        })
    return feedback


def run(run_date: date, interactive: bool = True) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": run_date.isoformat()}}
        result = app.invoke({"run_date": run_date, "interactive": interactive}, config)
        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            result = app.invoke(Command(resume=ask_for_feedback(payload)), config)
        return result


def main():
    parser = argparse.ArgumentParser(description="Generate the daily technical learning brief.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--no-feedback", action="store_true", help="skip the rating step")
    args = parser.parse_args()

    result = run(args.date, interactive=not args.no_feedback)
    print(result.get("brief_markdown", ""))


if __name__ == "__main__":
    main()
