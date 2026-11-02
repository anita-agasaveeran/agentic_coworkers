"""career.db reads and writes, and read-only access to Agent 1's history."""

import json
import sqlite3
from datetime import date

import pytest

import config
import memory

TODAY = date(2026, 11, 1)


@pytest.fixture
def conn(tmp_path):
    connection = memory.connect(tmp_path / "career.db")
    yield connection
    connection.close()


def opportunity(url, fit, deadline=None, status="open"):
    return {"url": url, "title": f"Role {url[-1]}", "type": "internship", "source": "Greenhouse", "org": "Org",
            "location": None, "deadline": deadline, "found_on": date(2026, 10, 30), "fit_score": fit, "status": status,
            "requirements": [{"skill": "rag", "level": 2, "required": True, "evidence": "RAG experience"}]}


def test_opportunities_round_trip(conn):
    memory.record_opportunities(conn, [
        opportunity("https://x.org/1", 0.5), opportunity("https://x.org/2", 0.9, deadline=date(2026, 12, 1)),
        opportunity("https://x.org/3", 0.7, deadline=date(2026, 10, 31)),     # deadline passed
        opportunity("https://x.org/4", 0.8, status="closed"),
    ])
    stored = memory.stored_opportunities(conn, TODAY)
    assert [o["url"] for o in stored] == ["https://x.org/2", "https://x.org/1"]
    assert stored[0]["deadline"] == date(2026, 12, 1) and stored[0]["found_on"] == date(2026, 10, 30)
    assert stored[0]["requirements"] == [{"skill": "rag", "level": 2, "required": True, "evidence": "RAG experience"}]
    assert memory.known_urls(conn) == {f"https://x.org/{i}" for i in range(1, 5)}
    assert memory.newest_opportunity_date(conn) == date(2026, 10, 30)

    memory.record_opportunities(conn, [{**opportunity("https://x.org/1", 0.6), "requirements": []}])
    assert memory.stored_opportunities(conn, TODAY)[1]["requirements"] == []   # requirements replaced


def test_session_history_and_readiness(conn):
    questions = [{"id": "2026-10-30-q1", "skill": "rag", "kind": "concept", "difficulty": 2, "prompt": "Why RAG?"},
                 {"id": "2026-10-30-q2", "skill": "sql", "kind": "concept", "difficulty": 1, "prompt": "Joins?"}]
    assessments = [{"question_id": "2026-10-30-q1", "action": "answered", "answer": "a", "score": 0.5,
                    "criteria_scores": {"c": 3}, "misconceptions": ["m"]},
                   {"question_id": "2026-10-30-q2", "action": "skipped", "answer": "", "score": None}]
    memory.record_session(conn, date(2026, 10, 30), questions, assessments)
    history = memory.assessment_history(conn)
    assert [(h["skill"], h["score"], h["session_date"]) for h in history] == [("rag", 0.5, date(2026, 10, 30))]
    assert memory.assessment_history(conn, before=date(2026, 10, 30)) == []
    assert memory.recent_questions(conn, "rag") == ["Why RAG?"]

    memory.record_readiness(conn, date(2026, 10, 30), {"rag": 0.4, "python": 0.7}, {"rag"})
    memory.record_readiness(conn, TODAY, {"rag": 0.6, "python": 0.7}, {"rag"})
    assert memory.skill_trend(conn) == {"rag": [("2026-10-30", 0.4), ("2026-11-01", 0.6)]}
    assert memory.previous_readiness(conn, TODAY) == {"date": date(2026, 10, 30), "readiness": {"rag": 0.4, "python": 0.7}}
    assert memory.previous_readiness(conn, date(2026, 10, 30)) == {}


def test_plan_tasks(conn):
    stored = memory.record_plan(conn, date(2026, 10, 30), [
        {"task": "A", "skill": "rag", "hours": 2, "due": date(2026, 11, 5), "done_when": "x"},
        {"task": "B", "skill": "sql", "hours": 1, "due": date(2026, 11, 2), "done_when": "y"},
    ])
    assert all(t["id"] for t in stored)
    assert [t["task"] for t in memory.open_plan_tasks(conn)] == ["B", "A"]
    memory.update_task_status(conn, TODAY, [{"id": stored[1]["id"], "status": "done"}])
    again = memory.record_plan(conn, TODAY, memory.open_plan_tasks(conn))    # carried over, not duplicated
    assert [t["task"] for t in again] == ["A"]
    assert conn.execute("SELECT COUNT(*) FROM plan_tasks").fetchone()[0] == 2


@pytest.mark.parametrize("topic, expected", [
    ("astrophysics", {"astrophysics"}),
    ("llm agents and multi-agent systems", {"llm_agents"}),
    ("cnn cosmology", {"astrophysics"}),
    ("langchain", {"langgraph"}),
    ("dual agn", set()),
])
def test_skills_for_topic(topic, expected):
    assert memory.skills_for_topic(topic, memory.alias_index(config.load_skills())) == expected


def test_agent1_reading_is_read_only(tmp_path):
    path = tmp_path / "history.db"
    agent1 = sqlite3.connect(path)
    agent1.executescript("""
        CREATE TABLE articles_seen (url TEXT PRIMARY KEY, title TEXT, source TEXT, published TEXT, topics TEXT,
                                    score REAL, shown_on TEXT);
        CREATE TABLE feedback (url TEXT, rating INTEGER, tags TEXT, given_on TEXT);
    """)
    agent1.executemany("INSERT INTO articles_seen VALUES (?, ?, 'S', NULL, ?, 1, ?)", [
        ("https://a/1", "Galaxy mergers", json.dumps(["galaxy mergers", "astrophysics"]), "2026-10-30"),
        ("https://a/2", "RAG at scale", json.dumps(["retrieval-augmented generation"]), "2026-10-31"),
    ])
    agent1.execute("INSERT INTO feedback VALUES ('https://a/1', 4, '[]', '2026-10-30')")
    agent1.commit()
    agent1.close()

    reading = memory.agent1_reading(config.load_skills(), path)
    assert reading == {
        "astrophysics": {"articles": 1, "avg_rating": 4.0, "recent": [{"title": "Galaxy mergers", "url": "https://a/1"}]},
        "rag": {"articles": 1, "avg_rating": None, "recent": [{"title": "RAG at scale", "url": "https://a/2"}]},
    }
    assert memory.agent1_reading(config.load_skills(), tmp_path / "missing.db") == {}
