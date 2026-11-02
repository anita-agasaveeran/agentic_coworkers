"""Long-term memory: career.db (opportunities, assessments, skill history, plan tasks),
read-only access to Agent 1's reading history, and a question vector store."""

import json
import logging
import re
import sqlite3
from datetime import date
from pathlib import Path

import config
from state import Assessment, Opportunity, PlanTask, Question, SkillReq

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS opportunities (
    url        TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    type       TEXT NOT NULL,
    source     TEXT,
    org        TEXT,
    location   TEXT,
    deadline   TEXT,            -- ISO date, NULL if not stated
    found_on   TEXT NOT NULL,
    fit_score  REAL,
    status     TEXT NOT NULL DEFAULT 'open'   -- open | closed | applied | dismissed
);

CREATE TABLE IF NOT EXISTS requirements (
    url        TEXT NOT NULL REFERENCES opportunities(url),
    skill      TEXT NOT NULL,
    level      INTEGER CHECK (level BETWEEN 1 AND 3),
    required   INTEGER NOT NULL,  -- 1 = required, 0 = preferred
    evidence   TEXT,
    PRIMARY KEY (url, skill)
);

CREATE TABLE IF NOT EXISTS questions (
    id           TEXT PRIMARY KEY,
    session_date TEXT NOT NULL,
    skill        TEXT NOT NULL,
    kind         TEXT NOT NULL,
    difficulty   INTEGER CHECK (difficulty BETWEEN 1 AND 3),
    prompt       TEXT NOT NULL,
    grounding    TEXT,
    parent_id    TEXT REFERENCES questions(id)
);

CREATE TABLE IF NOT EXISTS assessments (
    question_id     TEXT PRIMARY KEY REFERENCES questions(id),
    action          TEXT NOT NULL,   -- answered | skipped
    answer          TEXT,
    hints_used      INTEGER NOT NULL DEFAULT 0,
    criteria_scores TEXT,            -- JSON {criterion: 1-5}
    tests_passed    REAL,
    score           REAL,
    feedback        TEXT,
    misconceptions  TEXT             -- JSON list
);

CREATE TABLE IF NOT EXISTS posting_texts (
    url        TEXT PRIMARY KEY REFERENCES opportunities(url),
    text       TEXT NOT NULL,     -- the text requirements were extracted from (evaluation)
    fetched_on TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_history (
    skill        TEXT NOT NULL,
    session_date TEXT NOT NULL,
    readiness    REAL NOT NULL,
    assessed     INTEGER NOT NULL,   -- 0 = self-rated fallback
    PRIMARY KEY (skill, session_date)
);

CREATE TABLE IF NOT EXISTS plan_tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_on   TEXT NOT NULL,
    skill        TEXT NOT NULL,
    task         TEXT NOT NULL,
    resource_url TEXT,
    hours        REAL,
    due          TEXT,
    done_when    TEXT,
    status       TEXT NOT NULL DEFAULT 'open',   -- open | partial | done | dropped
    updated_on   TEXT
);
"""


def connect(db_path=None) -> sqlite3.Connection:
    config.DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path or config.CAREER_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --- Reads -------------------------------------------------------------------

def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, date) else value


def stored_opportunities(conn: sqlite3.Connection, today: date) -> list[Opportunity]:
    """Open opportunities whose deadline hasn't passed, with their requirements, best fit first."""
    rows = conn.execute(
        "SELECT * FROM opportunities WHERE status = 'open' AND (deadline IS NULL OR deadline >= ?) "
        "ORDER BY fit_score DESC, found_on DESC", (today.isoformat(),)).fetchall()
    requirements: dict[str, list] = {}
    for r in conn.execute("SELECT * FROM requirements ORDER BY url, required DESC, level DESC"):
        requirements.setdefault(r["url"], []).append(SkillReq(
            skill=r["skill"], level=r["level"], required=bool(r["required"]), evidence=r["evidence"]))
    return [Opportunity(
        url=r["url"], title=r["title"], type=r["type"], source=r["source"], org=r["org"],
        location=r["location"], deadline=_date(r["deadline"]), found_on=_date(r["found_on"]),
        fit_score=r["fit_score"], status=r["status"], requirements=requirements.get(r["url"], []),
    ) for r in rows]


def newest_opportunity_date(conn: sqlite3.Connection) -> date | None:
    """found_on of the most recently stored opportunity (None if there are none)."""
    return _date(conn.execute("SELECT MAX(found_on) FROM opportunities").fetchone()[0])


def known_urls(conn: sqlite3.Connection) -> set[str]:
    """Every opportunity URL already stored, open or not, so discovery skips them."""
    return {r[0] for r in conn.execute("SELECT url FROM opportunities")}


def assessment_history(conn: sqlite3.Connection, before: date | None = None) -> list[dict]:
    """Every answered assessment joined with its question: {skill, kind, difficulty, score, session_date}.

    `before` limits it to sessions earlier than that date.
    """
    rows = conn.execute(
        "SELECT q.id, q.skill, q.kind, q.difficulty, q.session_date, a.score FROM assessments a "
        "JOIN questions q ON q.id = a.question_id WHERE a.action = 'answered' AND a.score IS NOT NULL "
        "AND (? IS NULL OR q.session_date < ?) ORDER BY q.session_date, q.id",
        (_iso(before), _iso(before))).fetchall()
    return [{**dict(r), "session_date": _date(r["session_date"])} for r in rows]


def open_plan_tasks(conn: sqlite3.Connection) -> list[PlanTask]:
    """Plan tasks with status open or partial, oldest due date first."""
    rows = conn.execute("SELECT * FROM plan_tasks WHERE status IN ('open', 'partial') ORDER BY due, id")
    return [PlanTask(id=r["id"], task=r["task"], skill=r["skill"], resource_url=r["resource_url"],
                     hours=r["hours"], due=_date(r["due"]), done_when=r["done_when"], status=r["status"])
            for r in rows]


def recent_questions(conn: sqlite3.Connection, skill: str, limit: int = 10) -> list[str]:
    """Prompts of the latest questions on `skill`, for the generator's do-not-repeat list."""
    rows = conn.execute("SELECT prompt FROM questions WHERE skill = ? ORDER BY session_date DESC, id DESC LIMIT ?",
                        (skill, limit))
    return [r[0] for r in rows]


def skill_trend(conn: sqlite3.Connection) -> dict[str, list[tuple[str, float]]]:
    """skill -> [(session_date, readiness), ...] oldest first, for assessed skills (Progress section)."""
    trend: dict[str, list[tuple[str, float]]] = {}
    for r in conn.execute("SELECT skill, session_date, readiness FROM skill_history WHERE assessed = 1 "
                          "ORDER BY session_date"):
        trend.setdefault(r["skill"], []).append((r["session_date"], r["readiness"]))
    return trend


def previous_readiness(conn: sqlite3.Connection, before: date) -> dict:
    """{"date", "readiness"} from the latest session before `before` ({} if there is none)."""
    last = conn.execute("SELECT MAX(session_date) FROM skill_history WHERE session_date < ?",
                        (before.isoformat(),)).fetchone()[0]
    if last is None:
        return {}
    rows = conn.execute("SELECT skill, readiness FROM skill_history WHERE session_date = ?", (last,))
    return {"date": date.fromisoformat(last), "readiness": {r["skill"]: r["readiness"] for r in rows}}


def alias_index(skills: dict) -> dict[str, str]:
    """Lowercase alias or id (underscores as spaces) -> skill id, from skills.yaml."""
    index = {}
    for area in skills["areas"].values():
        for skill_id, aliases in area.items():
            index[skill_id] = skill_id
            index[skill_id.replace("_", " ")] = skill_id
            for alias in aliases or []:
                index[alias.lower()] = skill_id
    return index


def skills_for_topic(topic: str, index: dict[str, str]) -> set[str]:
    """Skill ids for a free-text topic: an exact alias match, else every alias found as whole words."""
    topic = topic.lower().strip()
    if topic in index:
        return {index[topic]}
    return {skill for alias, skill in index.items()
            if len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", topic)}


def agent1_reading(skills: dict, db_path=None) -> dict[str, dict]:
    """Agent 1 reading evidence per skill id: {"articles": n, "avg_rating": x | None, "recent": [...]}.

    Opens Agent 1's history.db read-only (sqlite URI mode=ro) and maps the topics of articles
    it showed me to skill ids through skills.yaml aliases. `recent` holds up to three
    {"title", "url"} for grounding questions. Returns {} if the database doesn't exist.
    """
    path = Path(db_path or config.AGENT1_HISTORY_DB)
    if not path.exists():
        return {}
    index = alias_index(skills)
    reading: dict[str, dict] = {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT a.url, a.title, a.topics, a.shown_on, AVG(f.rating) AS rating FROM articles_seen a "
                "LEFT JOIN feedback f USING (url) GROUP BY a.url ORDER BY a.shown_on DESC").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        log.warning("couldn't read Agent 1 history (%s)", exc)
        return {}
    for row in rows:
        matched = set().union(*(skills_for_topic(t, index) for t in json.loads(row["topics"] or "[]")))
        for skill in matched:
            entry = reading.setdefault(skill, {"articles": 0, "ratings": [], "recent": []})
            entry["articles"] += 1
            if row["rating"] is not None:
                entry["ratings"].append(row["rating"])
            if len(entry["recent"]) < 3:
                entry["recent"].append({"title": row["title"], "url": row["url"]})
    return {skill: {"articles": e["articles"], "recent": e["recent"],
                    "avg_rating": round(sum(e["ratings"]) / len(e["ratings"]), 2) if e["ratings"] else None}
            for skill, e in reading.items()}


# --- Writes ------------------------------------------------------------------

def record_opportunities(conn: sqlite3.Connection, opportunities: list[Opportunity]) -> None:
    """Upsert opportunities and replace their requirements."""
    with conn:
        for o in opportunities:
            conn.execute(
                "INSERT INTO opportunities (url, title, type, source, org, location, deadline, found_on, "
                "fit_score, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(url) DO UPDATE SET "
                "title = excluded.title, org = excluded.org, location = excluded.location, "
                "deadline = excluded.deadline, fit_score = excluded.fit_score, status = excluded.status",
                (o["url"], o["title"], o["type"], o.get("source"), o.get("org"), o.get("location"),
                 _iso(o.get("deadline")), _iso(o["found_on"]), o.get("fit_score"), o.get("status", "open")),
            )
            conn.execute("DELETE FROM requirements WHERE url = ?", (o["url"],))
            conn.executemany(
                "INSERT INTO requirements (url, skill, level, required, evidence) VALUES (?, ?, ?, ?, ?)",
                [(o["url"], r["skill"], r["level"], int(r["required"]), r["evidence"])
                 for r in o.get("requirements", [])],
            )


def record_posting_texts(conn: sqlite3.Connection, session_date: date, texts: dict[str, str]) -> None:
    """Store the posting text each kept opportunity's requirements were extracted from."""
    with conn:
        conn.executemany("INSERT OR REPLACE INTO posting_texts (url, text, fetched_on) VALUES (?, ?, ?)",
                         [(url, text, session_date.isoformat()) for url, text in texts.items()])


def posting_texts(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["url"]: r["text"] for r in conn.execute("SELECT url, text FROM posting_texts")}


def record_session(conn: sqlite3.Connection, session_date: date,
                   questions: list[Question], assessments: list[Assessment]) -> None:
    """Insert today's questions and assessments (replacing rows from a rerun of the same session)."""
    with conn:
        for q in questions:
            conn.execute(
                "INSERT OR REPLACE INTO questions (id, session_date, skill, kind, difficulty, prompt, grounding, "
                "parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (q["id"], session_date.isoformat(), q["skill"], q["kind"], q["difficulty"], q["prompt"],
                 q.get("grounding"), q.get("parent_id")),
            )
        for a in assessments:
            conn.execute(
                "INSERT OR REPLACE INTO assessments (question_id, action, answer, hints_used, criteria_scores, "
                "tests_passed, score, feedback, misconceptions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (a["question_id"], a["action"], a.get("answer"), a.get("hints_used", 0),
                 json.dumps(a.get("criteria_scores", {})), a.get("tests_passed"), a.get("score"),
                 a.get("feedback"), json.dumps(a.get("misconceptions", []))),
            )


def record_readiness(conn: sqlite3.Connection, session_date: date,
                     readiness: dict[str, float], assessed: set[str]) -> None:
    """Upsert one skill_history row per skill for this session."""
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO skill_history (skill, session_date, readiness, assessed) VALUES (?, ?, ?, ?)",
            [(skill, session_date.isoformat(), value, int(skill in assessed)) for skill, value in readiness.items()],
        )


def update_task_status(conn: sqlite3.Connection, session_date: date, updates: list[dict]) -> None:
    """Apply check-in results ({"id", "status"}) to plan_tasks."""
    with conn:
        conn.executemany("UPDATE plan_tasks SET status = ?, updated_on = ? WHERE id = ?",
                         [(u["status"], session_date.isoformat(), u["id"]) for u in updates])


def record_plan(conn: sqlite3.Connection, session_date: date, tasks: list[PlanTask]) -> list[PlanTask]:
    """Insert new plan tasks; returns them with their ids. Carried-over tasks keep their row."""
    stored = []
    with conn:
        for t in tasks:
            if t.get("id") is None:
                cursor = conn.execute(
                    "INSERT INTO plan_tasks (created_on, skill, task, resource_url, hours, due, done_when, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_date.isoformat(), t["skill"], t["task"], t.get("resource_url"), t.get("hours"),
                     _iso(t.get("due")), t.get("done_when"), t.get("status", "open")),
                )
                t = {**t, "id": cursor.lastrowid}
            stored.append(t)
    return stored


# --- Vector store (repeat questions) -----------------------------------------

def load_vector_store(embeddings, path=None):
    """InMemoryVectorStore loaded from VECTOR_STORE_PATH (empty if the file doesn't exist)."""
    from langchain_core.vectorstores import InMemoryVectorStore

    path = Path(path or config.VECTOR_STORE_PATH)
    if path.exists():
        return InMemoryVectorStore.load(str(path), embeddings)
    return InMemoryVectorStore(embeddings)


def max_similarity(store, text: str) -> float:
    """Highest similarity between `text` and any past question (0.0 if the store is empty)."""
    if not store.store:
        return 0.0
    return store.similarity_search_with_score(text, k=1)[0][1]


def add_questions(store, questions: list[Question], path=None) -> None:
    """Add today's question prompts (metadata: id, skill) and persist the store."""
    if not questions:
        return
    store.add_texts([q["prompt"] for q in questions],
                    metadatas=[{"id": q["id"], "skill": q["skill"]} for q in questions],
                    ids=[q["id"] for q in questions])
    config.DATA_DIR.mkdir(exist_ok=True)
    store.dump(str(path or config.VECTOR_STORE_PATH))
