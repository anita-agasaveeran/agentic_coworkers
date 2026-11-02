"""The full graph with a fake LLM and fake web tools (the code runner is real)."""

import sqlite3
from datetime import timedelta
from pathlib import Path

from fakes import CODING, SESSION_DATE, ScriptedAnswers

EXTRA = "https://example.org/my-posting"


def db_rows(env, sql):
    conn = sqlite3.connect(env["tmp"] / "data" / "career.db")
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def first_session(env, **kwargs):
    answers = ScriptedAnswers()
    result = env["run"](answers, SESSION_DATE, session_size=3, extra_postings=[EXTRA], **kwargs)
    return result, answers


def test_discovery_keeps_open_postings_and_maps_skills(env):
    result, _ = first_session(env)
    by_url = {o["url"]: o for o in result["opportunities"]}
    assert set(by_url) == {"https://job-boards.greenhouse.io/acme/jobs/1",
                           "https://jobs.lever.co/beta/0f6a2c1e-1b2c-4d5e-8f90-123456789abc", EXTRA}
    assert {o["url"]: o["status"] for o in result["rejected"]} == {
        "https://job-boards.greenhouse.io/gamma/jobs/2": "closed", "https://www.nsf.gov/funding/reu": "dismissed"}

    acme = by_url["https://job-boards.greenhouse.io/acme/jobs/1"]
    assert {r["skill"] for r in acme["requirements"]} == {"rag", "llm_agents", "python", "other:fortran"}
    assert acme["deadline"].isoformat() == "2026-12-15" and acme["org"] == "Acme"
    # priority 1.0 x (0.5 + 0.5 x overlap); overlap = (rag 0.5 + python 1) / (3 required + 0.5 preferred)
    assert acme["fit_score"] == round(1.0 * (0.5 + 0.5 * 1.5 / 3.5), 3)
    extra = by_url[EXTRA]
    assert extra["title"] == "Research Assistant, ML for Astronomy" and extra["type"] == "research"
    assert {r["skill"] for r in extra["requirements"]} == {"ml_for_science", "python"}
    assert [o["fit_score"] for o in result["opportunities"]] == sorted(
        (o["fit_score"] for o in result["opportunities"]), reverse=True)
    # the query for a type that isn't a target was dropped
    assert [q["type"] for q in result["queries"]] == ["internship", "research"]


def test_session_asks_grades_and_follows_up(env):
    result, answers = first_session(env)
    plan = result["session_plan"]
    assert len(plan) == 3 and {p["kind"] for p in plan} == {"coding", "scenario", "concept"}

    questions = result["questions"]
    assert [q["id"] for q in questions if not q["parent_id"]] == [f"2026-11-01-q{n}" for n in (1, 2, 3)]
    grades = {a["question_id"]: a for a in result["assessments"]}
    for q in questions:     # one follow-up exactly where the answer scored < 0.6 or >= 0.85
        if not q["parent_id"]:
            score = grades[q["id"]]["score"]
            follow_ups = [f["id"] for f in questions if f["parent_id"] == q["id"]]
            assert follow_ups == ([f"{q['id']}f"] if score < 0.6 or score >= 0.85 else [])

    coding = next(q for q in questions if q["kind"] == "coding")
    assert "test_broken" not in coding["tests"] and "test_small" in coding["tests"]
    assert Path(coding["workspace_file"]).read_text() == CODING["starter_code"]
    assert all("reference_answer" not in p["question"] for p in answers.payloads)
    assert grades[coding["id"]]["tests_passed"] == 1.0
    assert grades[coding["id"]]["score"] == round(0.6 * 1.0 + 0.4 * 0.75 - 0.1, 3)   # one hint used
    weak = next(a for a in result["assessments"] if a["answer"] == "weak answer")
    assert weak["score"] == 0.25 and weak["misconceptions"] == ["mixes up X and Y"]
    # the grade of each answer is shown with the next question
    assert answers.payloads[1]["previous"]["question_id"] == questions[0]["id"]


def test_report_plan_and_memory(env):
    result, _ = first_session(env)
    report = Path(result["report_path"]).read_text()
    for heading in ("# Career Readiness Report — Nov 1, 2026", "## Readiness by target", "## Opportunities",
                    "## What these roles ask for", "## Today's assessment", "## Strengths", "## Gaps",
                    "## Progress", "## 14-day preparation plan", "### References"):
        assert heading in report
    assert "fortran (outside taxonomy)" in report
    assert "Unsupported claim" not in report and "q99" not in report     # uncited finding / bad evidence dropped
    assert Path(result["transcript_path"]).read_text().count("Reference answer") == len(result["questions"])

    plan = result["prep_plan"]
    assert [t["task"] for t in plan] == ["Read the RAG guide", "Build an agent"]
    assert plan[0]["resource_url"].startswith("https://docs.langchain.com/")
    assert plan[1]["hours"] == 8.0 and plan[1]["due"] == SESSION_DATE + timedelta(days=14)
    assert all(t["id"] for t in plan)
    assert [r["url"] for r in result["resources"]] == [plan[0]["resource_url"]]

    assert len(db_rows(env, "SELECT * FROM opportunities")) == 5
    assert len(db_rows(env, "SELECT * FROM questions")) == len(result["questions"])
    assert db_rows(env, "SELECT COUNT(*) FROM plan_tasks WHERE status = 'open'") == [(2,)]
    assessed = dict(db_rows(env, "SELECT skill, assessed FROM skill_history"))
    assert assessed["python"] in (0, 1) and sum(assessed.values()) >= 2


def test_next_session_checks_in_and_reuses_opportunities(env):
    first_session(env)
    queries = env["llm"].count("OpportunityQueries")
    answers = ScriptedAnswers(text_answers=["strong answer", "strong answer"],
                              check_in={"Read the RAG guide": "done"})
    result = env["run"](answers, SESSION_DATE + timedelta(days=2), session_size=2)

    assert env["llm"].count("OpportunityQueries") == queries        # fresh: no rediscovery
    assert answers.payloads[0]["kind"] == "check_in" and len(answers.payloads[0]["tasks"]) == 2
    assert result["previous"]["date"] == SESSION_DATE
    report = result["report_markdown"]
    assert "since Nov 1, 2026" in report and "completed since the last session: 1 of 2" in report
    assert "Build an agent _(carried over)_" in report
    # day 1's two tasks: one done at the check-in, one carried over; plus day 2's two new tasks
    assert sorted(db_rows(env, "SELECT created_on, task, status FROM plan_tasks")) == [
        ("2026-11-01", "Build an agent", "open"), ("2026-11-01", "Read the RAG guide", "done"),
        ("2026-11-03", "Build an agent", "open"), ("2026-11-03", "Read the RAG guide", "open")]


def test_retest_of_a_weak_skill(env):
    first, _ = first_session(env)
    weak_question = next(a["question_id"] for a in first["assessments"] if a["answer"] == "weak answer")
    weak_skill = next(q["skill"] for q in first["questions"] if q["id"] == weak_question)
    result = env["run"](ScriptedAnswers(), SESSION_DATE + timedelta(days=1), session_size=3)
    assert result["weak_skills"] == [weak_skill]                # weak 0.25 + follow-up 0.75 -> mean 0.5
    retests = [p["skill"] for p in result["session_plan"] if p["reason"] == "retest"]
    assert retests in ([weak_skill], [])                       # [] when it is already a top gap
    assert weak_skill in [p["skill"] for p in result["session_plan"]]


def test_report_only_session(env):
    result = env["run"](ScriptedAnswers(), SESSION_DATE, session_size=0)
    assert result["session_plan"] == [] and result["questions"] == []
    assert "No questions this session." in result["report_markdown"]
    assert "transcript_path" not in result


def test_quit_and_skip(env):
    answers = ScriptedAnswers(text_answers=["skip", "quit"])
    result = env["run"](answers, SESSION_DATE, session_size=3)
    actions = [a["action"] for a in result["assessments"]]
    assert "skipped" in actions and result["quit"]
    assert "Session ended early" in result["report_markdown"]


def test_unusable_coding_question_becomes_a_concept_question(env):
    env["llm"].coding_tests = "from solution import add\n\ndef test_wrong():\n    assert add(1, 1) == 3\n"
    result, _ = first_session(env)
    assert not any(q["kind"] == "coding" for q in result["questions"])
    assert env["llm"].count("GeneratedQuestion") >= 5


def test_without_tavily_uses_stored_opportunities(env, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    result = env["run"](ScriptedAnswers(), SESSION_DATE, session_size=1)
    assert env["llm"].count("OpportunityQueries") == 0 and result["opportunities"] == []
    assert result["skill_demand"] == {"python": 1.0, "rag": 1.0}       # falls back to profile skills
    assert "No opportunities analyzed yet" in result["report_markdown"]
