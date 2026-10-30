"""Long-term memory: SQLite history (seen articles, feedback, topic progress) and a summary vector store."""

import sqlite3
from datetime import date

from config import DATA_DIR, HISTORY_DB, VECTOR_STORE_PATH
from state import Article, ArticleSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles_seen (
    url        TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    source     TEXT NOT NULL,
    published  TEXT,
    topics     TEXT,            -- JSON list
    score      REAL,
    shown_on   TEXT NOT NULL     -- run date (ISO)
);

CREATE TABLE IF NOT EXISTS feedback (
    url        TEXT NOT NULL REFERENCES articles_seen(url),
    rating     INTEGER CHECK (rating BETWEEN 1 AND 5),
    tags       TEXT,            -- JSON list, e.g. ["more", "too basic"]
    given_on   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_progress (
    topic      TEXT PRIMARY KEY,
    exposures  INTEGER NOT NULL DEFAULT 0,
    last_seen  TEXT,
    avg_rating REAL
);
"""


def connect(db_path=HISTORY_DB) -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --- Reads -------------------------------------------------------------------

def seen_urls(conn: sqlite3.Connection) -> set[str]:
    """All article URLs already shown in a past brief."""
    raise NotImplementedError


def learning_progress(conn: sqlite3.Connection, today: date, window_days: int = 14) -> dict:
    """Signals for personalization.

    Returns {"recent_topics": {topic: exposures}, "liked_topics": [...],
             "disliked_topics": [...], "under_covered": [...]}.
    """
    # TODO: under_covered = profile interests with few exposures in the window;
    #       liked/disliked from avg_rating and "more"/"less" feedback tags.
    raise NotImplementedError


# --- Writes ------------------------------------------------------------------

def record_brief(conn: sqlite3.Connection, run_date: date, selected: list[Article]) -> None:
    """Insert the selected articles into articles_seen and bump topic_progress exposures."""
    raise NotImplementedError


def record_feedback(conn: sqlite3.Connection, run_date: date, feedback: list[dict]) -> None:
    """Store ratings/tags and update topic_progress.avg_rating for the rated articles' topics."""
    raise NotImplementedError


# --- Vector store (novelty / near-duplicates) --------------------------------

def load_vector_store():
    """InMemoryVectorStore loaded from VECTOR_STORE_PATH (empty store if the file doesn't exist)."""
    # TODO: InMemoryVectorStore.load(path, get_embeddings()) or InMemoryVectorStore(get_embeddings())
    raise NotImplementedError


def max_similarity(store, text: str) -> float:
    """Highest similarity between `text` and any past summary (0.0 if the store is empty)."""
    raise NotImplementedError


def add_summaries(store, summaries: list[ArticleSummary]) -> None:
    """Add today's summaries (metadata: url) and persist the store with store.dump(VECTOR_STORE_PATH)."""
    raise NotImplementedError
