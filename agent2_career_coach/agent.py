"""nextLevelPersonalCoachForCareerLevel: LangGraph wiring and command-line entry point.

    python agent.py                         # coaching session; discovers opportunities if stale
    python agent.py --refresh               # force opportunity discovery
    python agent.py --questions 3           # shorter session
    python agent.py --add-posting <url>     # analyze a posting I found myself (repeatable)
    python agent.py --report-only           # report from stored data, no questions

An unfinished session for the same date (e.g. the terminal was closed mid-question) is resumed.
"""

import argparse
from datetime import date
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import config
import nodes
from state import CoachState

TASK_STATUSES = {"d": "done", "p": "partial", "o": "open", "x": "dropped", "": "open"}
COMMANDS = {"hint", "skip", "quit"}


def build_graph(checkpointer=None):
    graph = StateGraph(CoachState)

    graph.add_node("load_profile", nodes.load_profile)
    graph.add_node("check_in_plan", nodes.check_in_plan)
    graph.add_node("plan_opportunity_queries", nodes.plan_opportunity_queries)
    graph.add_node("discover_opportunities", nodes.discover_opportunities)
    graph.add_node("analyze_requirements", nodes.analyze_requirements)
    graph.add_node("assess_readiness", nodes.assess_readiness)
    graph.add_node("plan_session", nodes.plan_session)
    graph.add_node("generate_question", nodes.generate_question)
    graph.add_node("ask_and_collect_answer", nodes.ask_and_collect_answer)
    graph.add_node("grade_answer", nodes.grade_answer)
    graph.add_node("generate_follow_up", nodes.generate_follow_up)
    graph.add_node("summarize_strengths_gaps", nodes.summarize_strengths_gaps)
    graph.add_node("recommend_resources_and_plan", nodes.recommend_resources_and_plan)
    graph.add_node("compose_report", nodes.compose_report)
    graph.add_node("update_memory", nodes.update_memory)

    graph.add_edge(START, "load_profile")
    graph.add_edge("load_profile", "check_in_plan")
    graph.add_conditional_edges("check_in_plan", nodes.route_after_check_in)
    graph.add_edge("plan_opportunity_queries", "discover_opportunities")
    graph.add_edge("discover_opportunities", "analyze_requirements")
    graph.add_edge("analyze_requirements", "assess_readiness")
    graph.add_edge("assess_readiness", "plan_session")
    graph.add_conditional_edges("plan_session", nodes.route_after_plan)
    graph.add_edge("generate_question", "ask_and_collect_answer")
    graph.add_edge("generate_follow_up", "ask_and_collect_answer")
    graph.add_edge("ask_and_collect_answer", "grade_answer")
    graph.add_conditional_edges("grade_answer", nodes.route_after_grading)
    graph.add_edge("summarize_strengths_gaps", "recommend_resources_and_plan")
    graph.add_edge("recommend_resources_and_plan", "compose_report")
    graph.add_edge("compose_report", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile(checkpointer=checkpointer)


# --- Terminal prompts for the interrupts -------------------------------------

def ask_check_in(payload: dict) -> list[dict]:
    """Status of each open plan task from the last session."""
    print("\nPlan check-in: [d]one, [p]artial, [o]pen (Enter), or [x] drop")
    updates = []
    for task in payload["tasks"]:
        while (raw := input(f"  {task['task']} (due {task['due']}): ").strip().lower()) not in TASK_STATUSES:
            print("  Please enter d, p, o, x, or press Enter.")
        updates.append({"id": task["id"], "status": TASK_STATUSES[raw]})
    return updates


def read_text_answer() -> str:
    """Lines until an empty line (leading empty lines are ignored)."""
    lines = []
    while True:
        line = input()
        if not line.strip():
            if lines:
                return "\n".join(lines)
            continue
        lines.append(line)


def ask_question(payload: dict) -> dict:
    """Show a question, handle hint/skip/quit, and return the answer for grade_answer."""
    q = payload["question"]
    label = "Follow-up" if q.get("parent_id") else f"Question {payload['number']}/{payload['total']}"
    print(f"\n{label} · {q['skill']} · {q['kind']} · difficulty {q['difficulty']}\n\n{q['prompt']}\n")
    coding = q["kind"] == "coding"
    if coding:
        print(f"Write your solution in {q['workspace_file']}, then press Enter.")
    else:
        print("Type your answer and finish with an empty line.")
    print("Commands: hint, skip, quit")

    hints, used = q.get("hints") or [], 0
    while True:
        reply = input("> ").strip() if coding else read_text_answer()
        command = reply.strip().lower()
        if command in {"skip", "quit"}:
            return {"action": command, "answer": "", "hints_used": used}
        if command == "hint":
            if used < len(hints):
                print(f"Hint {used + 1}: {hints[used]}")
                used += 1
            else:
                print("No more hints.")
            continue
        answer = Path(q["workspace_file"]).read_text() if coding else reply
        if answer.strip():
            return {"action": "answer", "answer": answer, "hints_used": used}
        print("Empty answer. Type skip to skip this question.")


def answer_interrupt(payload: dict):
    return ask_check_in(payload) if payload["kind"] == "check_in" else ask_question(payload)


# --- Run ---------------------------------------------------------------------

def run(session_date: date, interactive: bool = True, refresh: bool = False,
        session_size: int = config.SESSION_QUESTIONS, extra_postings: list[str] | None = None) -> dict:
    config.DATA_DIR.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as checkpointer:
        app = build_graph(checkpointer)
        thread = {"configurable": {"thread_id": session_date.isoformat()}}
        snapshot = app.get_state(thread)
        if interactive and snapshot.next and snapshot.interrupts:
            print(f"Resuming the unfinished session for {session_date.isoformat()}.")
            result = app.invoke(Command(resume=answer_interrupt(snapshot.interrupts[0].value)), thread)
        else:
            result = app.invoke({
                "session_date": session_date,
                "interactive": interactive,
                "refresh": refresh,
                "session_size": session_size,
                "extra_postings": extra_postings or [],
            }, thread)
        while result.get("__interrupt__"):
            result = app.invoke(Command(resume=answer_interrupt(result["__interrupt__"][0].value)), thread)
        return result


def main():
    parser = argparse.ArgumentParser(description="Run a career coaching session.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--refresh", action="store_true", help="force opportunity discovery")
    parser.add_argument("--questions", type=int, default=config.SESSION_QUESTIONS, help="questions this session")
    parser.add_argument("--add-posting", action="append", default=[], metavar="URL",
                        help="analyze a posting found elsewhere (repeatable)")
    parser.add_argument("--report-only", action="store_true", help="report from stored data, no questions")
    args = parser.parse_args()

    result = run(args.date, refresh=args.refresh, extra_postings=args.add_posting,
                 session_size=0 if args.report_only else args.questions)
    print(result.get("report_markdown", ""))


if __name__ == "__main__":
    main()
