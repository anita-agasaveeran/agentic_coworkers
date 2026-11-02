"""Scoring (ARCHITECTURE.md §7), session planning, and small node helpers."""

from datetime import date, timedelta

import pytest

import config
import memory
import nodes
from state import CriterionScore, Grade

TODAY = date(2026, 11, 1)


def opp(type_, fit, *reqs):
    return {"type": type_, "fit_score": fit,
            "requirements": [{"skill": s, "level": lvl, "required": r, "evidence": ""} for s, lvl, r in reqs]}


@pytest.mark.parametrize("kind, criteria, tests, hints, expected", [
    ("concept", {"a": 5, "b": 5}, None, 0, 1.0),
    ("concept", {"a": 3, "b": 1}, None, 0, 0.25),
    ("concept", {"a": 5}, None, 2, 0.8),
    ("coding", {"a": 3}, 1.0, 0, 0.8),          # 0.6 x 1.0 + 0.4 x 0.5
    ("coding", {"a": 5}, 0.0, 0, 0.4),
    ("coding", {"a": 1}, 0.0, 3, 0.0),          # never below 0
    ("concept", {}, None, 0, 0.0),
])
def test_answer_score(kind, criteria, tests, hints, expected):
    assert nodes.answer_score(kind, criteria, tests, hints) == expected


def test_readiness_is_recency_weighted_and_scaled_by_difficulty():
    history = [
        {"skill": "rag", "difficulty": 3, "score": 1.0, "session_date": TODAY},
        {"skill": "rag", "difficulty": 3, "score": 0.0,
         "session_date": TODAY - timedelta(days=config.READINESS_HALF_LIFE_DAYS)},
        {"skill": "python", "difficulty": 1, "score": 1.0, "session_date": TODAY},
    ]
    readiness, assessed = nodes.readiness_from_history(
        history, {"python": "beginner", "sql": "Intermediate (rusty)", "git": "expert"},
        {"sql": {"articles": 10}, "astrophysics": {"articles": 1}}, TODAY)
    assert readiness["rag"] == round(1.0 / 1.5, 3)            # the older 0 counts half
    assert readiness["python"] == 0.6                         # assessed beats self-rated; easy questions cap it
    assert readiness["sql"] == 0.5 + config.READING_BONUS      # self-rated + full reading bonus
    assert readiness["astrophysics"] == round(config.READING_BONUS / config.READING_FULL_AT, 3)
    assert "git" not in readiness                             # unknown level word
    assert assessed == {"rag", "python"}


def test_skill_demand_and_targets():
    demand, targets = nodes.skill_demand([
        opp("internship", 1.0, ("rag", 2, True), ("python", 1, False)),
        opp("internship", 0.5, ("rag", 3, True), ("sql", 1, True)),
    ])
    assert demand == {"rag": 1.0, "python": round(0.5 / 1.5, 3), "sql": round(0.5 / 1.5, 3)}
    assert targets == {"rag": 3, "python": 1, "sql": 1}
    assert nodes.skill_demand([]) == ({}, {})


def test_skill_gaps_skip_other_skills_and_sort():
    gaps = nodes.skill_gaps({"rag": 1.0, "python": 1.0, "other:fortran": 1.0},
                            {"rag": 3, "python": 1}, {"rag": 0.35, "python": 0.9})
    assert gaps == {"rag": 0.5, "python": 0.0}


def test_role_readiness():
    opportunities = [opp("internship", 1.0, ("rag", 2, True), ("python", 1, True)),
                     opp("research", 1.0, ("astrophysics", 2, True))]
    result = nodes.role_readiness({"type": "internship"}, opportunities, {"rag": 0.35, "python": 0.9})
    assert result == {"readiness": 0.75, "boost": ["rag"]}     # mean of 0.35/0.7 and 1
    assert nodes.role_readiness({"type": "job"}, opportunities, {}) == {"readiness": None, "boost": []}


@pytest.mark.parametrize("readiness, expected", [(0.0, 1), (0.39, 1), (0.4, 2), (0.69, 2), (0.7, 3), (1.0, 3)])
def test_difficulty_for(readiness, expected):
    assert nodes.difficulty_for(readiness) == expected


def plan_state(demand, readiness, n=5, weak=()):
    return {"interactive": True, "session_size": n, "skill_demand": demand,
            "skill_targets": {s: 2 for s in demand}, "readiness": readiness, "weak_skills": list(weak),
            "profile": {"skills": {}}}


def test_plan_session_covers_gaps_kinds_and_a_retest():
    demand = {"llm_agents": 1.0, "rag": 0.9, "system_design": 0.8, "python": 0.5, "sql": 0.4, "git": 0.3}
    # gaps: llm_agents 0.7, system_design 0.56, sql 0.28, git 0.21, rag 0.18, python 0
    plan = nodes.plan_session(plan_state(demand, {"python": 0.95, "rag": 0.5}, weak=["python"]))["session_plan"]
    assert [p["skill"] for p in plan] == ["llm_agents", "system_design", "sql", "git", "python"]
    assert plan[-1]["reason"] == "retest"
    assert [p["kind"] for p in plan] == ["scenario", "concept", "concept", "concept", "coding"]
    assert [p["difficulty"] for p in plan] == [1, 1, 1, 1, 3]
    # a weak skill that is already a top gap isn't asked twice
    plan = nodes.plan_session(plan_state(demand, {"rag": 0.5}, weak=["llm_agents"]))["session_plan"]
    assert [p["reason"] for p in plan] == ["gap"] * 5


def test_plan_session_adds_a_coding_skill_when_no_gap_is_codable():
    plan = nodes.plan_session(plan_state({"system_design": 1.0, "cloud_platforms": 1.0}, {}, n=3))["session_plan"]
    assert [(p["skill"], p["kind"]) for p in plan] == [
        ("system_design", "scenario"), ("cloud_platforms", "concept"), ("python", "coding")]


def test_plan_session_report_only():
    assert nodes.plan_session(plan_state({"rag": 1.0}, {}, n=0)) == {"session_plan": []}
    assert nodes.plan_session({**plan_state({"rag": 1.0}, {}), "interactive": False}) == {"session_plan": []}


@pytest.mark.parametrize("raw, expected", [
    ("Python", "python"), ("retrieval-augmented generation", "rag"), ("machine_learning", "machine_learning"),
    ("ML for astronomy", "ml_for_science"), ("Fortran 90", "other:fortran 90"), ("other:Rust", "other:rust"),
])
def test_normalize_skill(raw, expected):
    index = memory.alias_index(config.load_skills())
    assert nodes.normalize_skill(raw, index) == expected


def test_fit_score():
    profile = {"targets": [{"type": "internship", "priority": 0.8}], "skills": {"python": "advanced", "rag": "beginner"}}
    reqs = [{"skill": "python", "required": True}, {"skill": "rag", "required": True},
            {"skill": "sql", "required": False}]
    assert nodes.fit_score(reqs, "internship", profile) == round(0.8 * (0.5 + 0.5 * 1.5 / 2.5), 3)
    assert nodes.fit_score(reqs, "research", profile) == 0.0


def test_drop_tests():
    tests = ("from solution import f\n\ndef test_a():\n    assert f()\n\ndef test_b():\n    assert not f()\n\n"
             "class TestC:\n    def test_b(self):\n        pass\n")
    kept = nodes.drop_tests(tests, {"test_b"})
    assert "def test_a" in kept and "def test_b" not in kept and "class TestC" in kept


def test_match_criteria_by_name_then_position():
    grade = Grade(criteria=[CriterionScore(criterion="Clarity", score=4), CriterionScore(criterion="other", score=9)],
                  feedback="", misconceptions=[])
    assert nodes.match_criteria(["correctness", "clarity"], grade) == {"correctness": 4, "clarity": 4}
    assert nodes.match_criteria(["a", "b", "c"], grade) == {"a": 4, "b": 5}


@pytest.mark.parametrize("skill, expected", [
    ("rag", "rag retrieval-augmented generation tutorial"),
    ("git", "git git and version control tutorial"),
    ("other:fortran", "other:fortran tutorial"),
])
def test_resource_query(skill, expected):
    assert nodes.resource_query(skill, config.load_skills()) == expected


@pytest.mark.parametrize("task, expected", [
    ("Carry-over: read the RAG guide and take notes.", True),
    ("Read the RAG guide and take notes", True),
    ("Solve 10 interval problems", False),
])
def test_restates_a_carried_over_task(task, expected):
    assert nodes.restates(task, [{"task": "Read the RAG guide and take notes."}]) is expected
