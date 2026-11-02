import sys

import pytest

import agent
import config
import nodes
import tools
from agent import build_graph
from memory import connect
from state import OpportunityType

EXPECTED_NODES = {
    "load_profile", "check_in_plan", "plan_opportunity_queries", "discover_opportunities",
    "analyze_requirements", "assess_readiness", "plan_session", "generate_question",
    "ask_and_collect_answer", "grade_answer", "generate_follow_up", "summarize_strengths_gaps",
    "recommend_resources_and_plan", "compose_report", "update_memory",
}
OPPORTUNITY_TYPES = set(OpportunityType.__args__)


def question(qid, parent=None):
    return {"id": qid, "parent_id": parent}


def answered(score):
    return {"action": "answered", "score": score}


def test_graph_compiles_with_all_nodes():
    graph = build_graph().get_graph()
    assert EXPECTED_NODES <= set(graph.nodes)


def test_route_after_check_in():
    assert nodes.route_after_check_in({"discovery_due": True}) == "plan_opportunity_queries"
    assert nodes.route_after_check_in({"extra_postings": ["https://x.org/job"]}) == "analyze_requirements"
    assert nodes.route_after_check_in({}) == "assess_readiness"


def test_route_after_plan_skips_questions_for_report_only():
    assert nodes.route_after_plan({"session_plan": [{}]}) == "generate_question"
    assert nodes.route_after_plan({"session_plan": []}) == "summarize_strengths_gaps"


@pytest.mark.parametrize("score, expected", [
    (0.3, "generate_follow_up"),        # probe a weak answer
    (0.9, "generate_follow_up"),        # challenge a strong one
    (0.7, "generate_question"),         # in between: next slot
])
def test_route_after_grading_follow_up_rules(score, expected):
    state = {"session_plan": [{}, {}], "questions": [question("q1")], "assessments": [answered(score)]}
    assert nodes.route_after_grading(state) == expected


def test_route_after_grading_limits_follow_ups_and_ends_session():
    two_slots = [{}, {}]
    after_follow_up = {"session_plan": two_slots, "questions": [question("q1"), question("q1f", "q1")],
                       "assessments": [answered(0.3), answered(0.2)]}
    assert nodes.route_after_grading(after_follow_up) == "generate_question"
    skipped = {"session_plan": two_slots, "questions": [question("q1")], "assessments": [{"action": "skipped"}]}
    assert nodes.route_after_grading(skipped) == "generate_question"
    last_slot = {"session_plan": two_slots, "questions": [question("q1"), question("q2")],
                 "assessments": [answered(0.7), answered(0.7)]}
    assert nodes.route_after_grading(last_slot) == "summarize_strengths_gaps"
    assert nodes.route_after_grading({"quit": True, "session_plan": two_slots}) == "summarize_strengths_gaps"


def test_check_in_skipped_without_tasks_or_input():
    assert nodes.check_in_plan({"interactive": False, "open_tasks": [{"id": 1, "task": "t"}]}) == {"plan_updates": []}
    assert nodes.check_in_plan({"interactive": True, "open_tasks": []}) == {"plan_updates": []}


def test_config_files_load_and_agree():
    profile = config.load_profile_file()
    taxonomy = {s for area in config.load_skills()["areas"].values() for s in area}
    assert {t["type"] for t in profile["targets"]} <= OPPORTUNITY_TYPES
    assert set(profile["skills"]) <= taxonomy
    sources = config.load_opportunity_sources()
    covered = {t for s in sources["opportunity_sources"] for t in s["types"]}
    assert covered == OPPORTUNITY_TYPES
    assert sources["resource_sources"]


def test_skill_aliases_are_unique():
    names = []
    for area in config.load_skills()["areas"].values():
        for skill, aliases in area.items():
            names += [skill, *aliases]
    assert len(names) == len(set(names))


def test_career_schema(tmp_path):
    conn = connect(tmp_path / "career.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"opportunities", "requirements", "questions", "assessments", "skill_history", "plan_tasks"} <= tables


def test_agent1_tools_load_without_replacing_our_state_module():
    import state

    module = tools.agent1_tools()
    assert callable(module.load_page) and callable(module.url_resolves) and callable(module.search_arxiv)
    assert sys.modules["state"] is state


def test_cli_question_hint_then_answer(monkeypatch):
    replies = iter(["hint", "", "A trie stores keys by prefix.", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    payload = {"number": 1, "total": 5, "question": {
        "skill": "algorithms_data_structures", "kind": "concept", "difficulty": 1,
        "prompt": "What is a trie?", "hints": ["Think prefixes."], "parent_id": None}}
    assert agent.ask_question(payload) == {"action": "answer", "answer": "A trie stores keys by prefix.", "hints_used": 1}


def test_cli_coding_answer_reads_workspace_file(monkeypatch, tmp_path):
    solution = tmp_path / "q1.py"
    solution.write_text("def f():\n    return 1\n")
    replies = iter([""])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    payload = {"number": 2, "total": 5, "question": {
        "skill": "python", "kind": "coding", "difficulty": 2, "prompt": "Write f.",
        "hints": [], "parent_id": None, "workspace_file": str(solution)}}
    assert agent.ask_question(payload)["answer"] == solution.read_text()


def test_cli_check_in(monkeypatch):
    replies = iter(["d", "maybe", "", "x"])
    monkeypatch.setattr("builtins.input", lambda *a: next(replies))
    tasks = [{"id": 1, "task": "a", "due": "2026-11-10"}, {"id": 2, "task": "b", "due": "2026-11-12"},
             {"id": 3, "task": "c", "due": "2026-11-14"}]
    assert agent.ask_check_in({"tasks": tasks}) == [
        {"id": 1, "status": "done"}, {"id": 2, "status": "open"}, {"id": 3, "status": "dropped"}]
