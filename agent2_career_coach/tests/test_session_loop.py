"""The question loop through the real graph and CLI, with the unimplemented nodes replaced by stand-ins."""

from datetime import date

import pytest

import agent
import config
import nodes

SESSION_DATE = date(2026, 11, 1)


@pytest.fixture
def stub_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CHECKPOINT_DB", tmp_path / "data" / "checkpoints.db")
    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")

    def generate(state, parent=None):
        questions = state["questions"]
        n = len(nodes.top_level_questions(questions)) + (0 if parent else 1)
        new = {"id": f"q{n}" + ("f" if parent else ""), "parent_id": parent, "skill": "python",
               "kind": "concept", "difficulty": 1, "prompt": f"Question {n}", "hints": []}
        return {"questions": questions + [new]}

    def grade(state):
        reply = state["pending_answer"]
        if reply["action"] == "quit":
            return {"quit": True}
        return {"assessments": state["assessments"] + [{"question_id": state["questions"][-1]["id"], "action": "answered",
                                                        "score": 0.3, "answer": reply["answer"], "feedback": "ok"}]}

    stand_ins = {
        "load_profile": lambda s: {"open_tasks": [], "discovery_due": False, "questions": [],
                                   "assessments": [], "quit": False},
        "assess_readiness": lambda s: {},
        "plan_session": lambda s: {"session_plan": [{"skill": "python"}] * s["session_size"]},
        "generate_question": generate,
        "generate_follow_up": lambda s: generate(s, parent=s["questions"][-1]["id"]),
        "grade_answer": grade,
        "summarize_strengths_gaps": lambda s: {},
        "recommend_resources_and_plan": lambda s: {},
        "compose_report": lambda s: {"report_markdown": "report"},
        "update_memory": lambda s: {},
    }
    for name, fn in stand_ins.items():
        monkeypatch.setattr(nodes, name, fn)


def feed_input(monkeypatch, lines):
    """Replies to input(); a (question id, text) pair writes text as the answer, then presses Enter."""
    replies = iter(lines)

    def fake_input(*_):
        reply = next(replies)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, tuple):
            qid, text = reply
            path = agent.answer_file({"id": qid})
            path.write_text(path.read_text() + text)
            return ""
        return reply

    monkeypatch.setattr("builtins.input", fake_input)


def test_weak_answers_get_one_follow_up_each(stub_nodes, monkeypatch):
    feed_input(monkeypatch, [("q1", "a1"), ("q1f", "a1f"), ("q2", "a2"), ("q2f", "a2f")])
    result = agent.run(SESSION_DATE, session_size=2)
    assert [q["id"] for q in result["questions"]] == ["q1", "q1f", "q2", "q2f"]
    assert [a["answer"] for a in result["assessments"]] == ["a1", "a1f", "a2", "a2f"]
    assert result["report_markdown"] == "report"


def test_quit_ends_the_session_with_a_report(stub_nodes, monkeypatch):
    feed_input(monkeypatch, [("q1", "a1"), "quit"])
    result = agent.run(SESSION_DATE, session_size=3)
    assert len(result["assessments"]) == 1 and result["quit"]
    assert result["report_markdown"] == "report"


def test_unfinished_session_is_resumed(stub_nodes, monkeypatch):
    feed_input(monkeypatch, [("q1", "a1"), KeyboardInterrupt()])   # terminal closed at the follow-up
    with pytest.raises(KeyboardInterrupt):
        agent.run(SESSION_DATE, session_size=1)
    feed_input(monkeypatch, [("q1f", "a1f")])
    result = agent.run(SESSION_DATE, session_size=1)
    assert [a["answer"] for a in result["assessments"]] == ["a1", "a1f"]
