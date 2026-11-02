"""nextLevelPersonalCoachForCareerLevel: LangGraph wiring and command-line entry point.

    python agent.py                         # coaching session; discovers opportunities if stale
    python agent.py --refresh               # force opportunity discovery
    python agent.py --questions 3           # shorter session
    python agent.py --add-posting <url>     # analyze a posting I found myself (repeatable)
    python agent.py --report-only           # report from stored data, no questions

An unfinished session for the same date (e.g. the terminal was closed mid-question) is resumed.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

try:
    import termios
except ImportError:      # Windows
    termios = None
try:
    import readline      # noqa: F401  input() then reads lines past the terminal's ~1024-character limit (macOS)
except ImportError:
    pass

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import config
import nodes
from state import CoachState

TASK_STATUSES = {"d": "done", "p": "partial", "o": "open", "x": "dropped", "": "open"}
COMMANDS = {"hint", "skip", "quit"}
ANSWER_MARKER = "<!-- Write your answer below this line, save the file, then press Enter in the terminal. -->"


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

def discard_typeahead() -> None:
    """Drop keys typed while the agent was working, so a stray Enter doesn't submit an answer."""
    if termios and sys.stdin.isatty():
        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def ask_check_in(payload: dict) -> list[dict]:
    """Status of each open plan task from the last session."""
    print("\nPlan check-in: [d]one, [p]artial, [o]pen (Enter), or [x] drop")
    discard_typeahead()
    updates = []
    for task in payload["tasks"]:
        while (raw := input(f"  {task['task']} (due {task['due']}): ").strip().lower()) not in TASK_STATUSES:
            print("  Please enter d, p, o, x, or press Enter.")
        updates.append({"id": task["id"], "status": TASK_STATUSES[raw]})
    return updates


def answer_file(question: dict) -> Path:
    """Workspace file for a text answer: the question, then my answer below ANSWER_MARKER."""
    return config.WORKSPACE_DIR / f"{question['id']}.md"


def prepare_answer_file(question: dict) -> Path:
    """Create the answer file; an existing one is kept, so a resumed session keeps a half-written answer."""
    path = answer_file(question)
    if not path.exists():
        config.WORKSPACE_DIR.mkdir(exist_ok=True)
        path.write_text(f"# {question['id']} · {question['skill']} · {question['kind']}\n\n"
                        f"{question['prompt']}\n\n{ANSWER_MARKER}\n\n")
    return path


def read_answer(question: dict) -> str:
    """My answer from the workspace file; empty while a coding file still has only the starter code."""
    if question["kind"] == "coding":
        text = Path(question["workspace_file"]).read_text()
        return "" if text.strip() == (question.get("starter_code") or nodes.EMPTY_STARTER).strip() else text
    return answer_file(question).read_text().split(ANSWER_MARKER, 1)[-1].strip()


def show_grade(grade: dict | None) -> None:
    """Feedback on the previous answer, shown before the next question."""
    if not grade or grade.get("action") != "answered":
        return
    tests = f" · tests {round(grade['tests_passed'] * 100)}%" if grade.get("tests_passed") is not None else ""
    print(f"\nScore {grade['score']:.2f}{tests}. {grade['feedback']}")
    for m in grade.get("misconceptions") or []:
        print(f"  - {m}")


def ask_question(payload: dict) -> dict:
    """Show a question, handle hint/skip/quit, and return the answer for grade_answer."""
    show_grade(payload.get("previous"))
    q = payload["question"]
    label = "Follow-up" if q.get("parent_id") else f"Question {payload['number']}/{payload['total']}"
    print(f"\n{label} · {q['skill']} · {q['kind']} · difficulty {q['difficulty']}\n\n{q['prompt']}\n")
    coding = q["kind"] == "coding"
    path = Path(q["workspace_file"]) if coding else prepare_answer_file(q)
    print(f"Write your {'solution' if coding else 'answer'} in {path}, save it, then press Enter.")
    print("Commands: hint, skip, quit")
    discard_typeahead()

    hints, used = q.get("hints") or [], 0
    while True:
        command = input("> ").strip().lower()
        if command in {"skip", "quit"}:
            return {"action": command, "answer": "", "hints_used": used}
        if command == "hint":
            if used < len(hints):
                print(f"Hint {used + 1}: {hints[used]}")
                used += 1
            else:
                print("No more hints.")
            continue
        if command:
            print(f"Write the answer in {path.name}, then press Enter. Commands: hint, skip, quit")
            continue
        answer = read_answer(q)
        if answer.strip():
            return {"action": "answer", "answer": answer, "hints_used": used}
        print("The file still has only the starter code. Save your solution and press Enter, or type skip." if coding
              else f"No answer below the marker line in {path.name} yet. Save it and press Enter, or type skip.")


def answer_interrupt(payload: dict):
    return ask_check_in(payload) if payload["kind"] == "check_in" else ask_question(payload)


# --- Run ---------------------------------------------------------------------

class SessionExists(Exception):
    """A finished session is saved for this date; a second one would overwrite its questions and report."""


def run(session_date: date, interactive: bool = True, refresh: bool = False,
        session_size: int = config.SESSION_QUESTIONS, extra_postings: list[str] | None = None) -> dict:
    config.DATA_DIR.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as checkpointer:
        app = build_graph(checkpointer)
        thread = {"configurable": {"thread_id": session_date.isoformat()}}
        snapshot = app.get_state(thread)
        if snapshot.values and not snapshot.next:
            raise SessionExists(f"The session for {session_date.isoformat()} is already finished "
                                f"({snapshot.values.get('report_path')}). Use --date for another day.")
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3"):     # the Anthropic SDK logs requests as httpx2
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        result = run(args.date, refresh=args.refresh, extra_postings=args.add_posting,
                     session_size=0 if args.report_only else args.questions)
    except SessionExists as exc:
        raise SystemExit(str(exc))
    print(result.get("report_markdown", ""))
    for key in ("report_path", "transcript_path"):
        if result.get(key):
            print(f"Saved {result[key]}")


if __name__ == "__main__":
    main()
