"""Long-term memory: career.db (opportunities, assessments, skill history, plan tasks),
read-only access to Agent 1's reading history, and a question vector store."""

import sqlite3
from datetime import date

import config
from state import Assessment, Opportunity, PlanTask, Question

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

def stored_opportunities(conn: sqlite3.Connection, today: date) -> list[Opportunity]:
    """Open opportunities whose deadline hasn't passed, with their requirements, best fit first."""
    raise NotImplementedError


def newest_opportunity_date(conn: sqlite3.Connection) -> date | None:
    """found_on of the most recently stored opportunity (None if there are none)."""
    raise NotImplementedError


def known_urls(conn: sqlite3.Connection) -> set[str]:
    """Every opportunity URL already stored, open or not, so discovery skips them."""
    raise NotImplementedError


def assessment_history(conn: sqlite3.Connection) -> list[dict]:
    """Every answered assessment joined with its question: {skill, kind, difficulty, score, session_date}."""
    raise NotImplementedError


def open_plan_tasks(conn: sqlite3.Connection) -> list[PlanTask]:
    """Plan tasks with status open or partial, oldest due date first."""
    raise NotImplementedError


def recent_questions(conn: sqlite3.Connection, skill: str, limit: int = 10) -> list[str]:
    """Prompts of the latest questions on `skill`, for the generator's do-not-repeat list."""
    raise NotImplementedError


def skill_trend(conn: sqlite3.Connection) -> dict[str, list[tuple[str, float]]]:
    """skill -> [(session_date, readiness), ...] oldest first, for the Progress section."""
    raise NotImplementedError


def agent1_reading(skills: dict, db_path=None) -> dict[str, dict]:
    """Agent 1 reading evidence per skill id: {"articles": n, "avg_rating": x}.

    Opens Agent 1's history.db read-only (sqlite URI mode=ro) and maps its topic_progress
    topics to skill ids through skills.yaml aliases. Returns {} if the database doesn't exist.
    """
    raise NotImplementedError


# --- Writes ------------------------------------------------------------------

def record_opportunities(conn: sqlite3.Connection, opportunities: list[Opportunity]) -> None:
    """Upsert opportunities and replace their requirements."""
    raise NotImplementedError


def record_session(conn: sqlite3.Connection, session_date: date,
                   questions: list[Question], assessments: list[Assessment]) -> None:
    """Insert today's questions and assessments."""
    raise NotImplementedError


def record_readiness(conn: sqlite3.Connection, session_date: date,
                     readiness: dict[str, float], assessed: set[str]) -> None:
    """Upsert one skill_history row per skill for this session."""
    raise NotImplementedError


def update_task_status(conn: sqlite3.Connection, session_date: date, updates: list[dict]) -> None:
    """Apply check-in results ({"id", "status"}) to plan_tasks."""
    raise NotImplementedError


def record_plan(conn: sqlite3.Connection, session_date: date, tasks: list[PlanTask]) -> list[PlanTask]:
    """Insert new plan tasks; returns them with their ids. Carried-over tasks keep their row."""
    raise NotImplementedError


# --- Vector store (repeat questions) -----------------------------------------

def load_vector_store(embeddings, path=None):
    """InMemoryVectorStore loaded from VECTOR_STORE_PATH (empty if the file doesn't exist)."""
    raise NotImplementedError


def max_similarity(store, text: str) -> float:
    """Highest similarity between `text` and any past question (0.0 if the store is empty)."""
    raise NotImplementedError


def add_questions(store, questions: list[Question], path=None) -> None:
    """Add today's question prompts (metadata: id, skill) and persist the store."""
    raise NotImplementedError
