"""Long-term memory: SQLite history (seen articles, feedback, topic progress) and a summary vector store."""

import json
import re
import sqlite3
from collections import Counter
from datetime import date, timedelta

import config
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

LIKED_RATING = 4
DISLIKED_RATING = 2
_STOPWORDS = {"and", "for", "the", "with", "systems", "system"}


def connect(db_path=None) -> sqlite3.Connection:
    config.DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path or config.HISTORY_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _keywords(text: str) -> set[str]:
    """Lowercase keywords with a trailing plural 's' removed ("agents" -> "agent")."""
    words = (w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS)
    return {w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w for w in words}


def _acronym(text: str) -> str | None:
    """Initials of a 2-4 word phrase ("retrieval-augmented generation" -> "rag")."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "".join(w[0] for w in words) if 2 <= len(words) <= 4 else None


def interest_matches_topic(interest: str, topic: str) -> bool:
    """Loose match between a profile interest and an LLM topic label (shared keyword or acronym)."""
    return bool((_keywords(interest) | {_acronym(interest)}) & _keywords(topic))


# --- Reads -------------------------------------------------------------------

def seen_urls(conn: sqlite3.Connection) -> set[str]:
    """All article URLs already shown in a past brief."""
    return {row["url"] for row in conn.execute("SELECT url FROM articles_seen")}


def learning_progress(conn: sqlite3.Connection, today: date, interests: dict[str, float],
                      window_days: int = 14) -> dict:
    """Signals for personalization.

    Returns {"recent_topics": {topic: count}, "liked_topics": [...], "disliked_topics": [...],
             "under_covered": [interest, ...]} where under_covered lists profile interests with
    no matching topic in the window, highest weight first.
    """
    start = (today - timedelta(days=window_days)).isoformat()
    recent = Counter()
    for row in conn.execute("SELECT topics FROM articles_seen WHERE shown_on >= ? AND shown_on < ?",
                            (start, today.isoformat())):
        recent.update(json.loads(row["topics"] or "[]"))

    liked, disliked = set(), set()
    for row in conn.execute("SELECT topic, avg_rating FROM topic_progress WHERE avg_rating IS NOT NULL"):
        if row["avg_rating"] >= LIKED_RATING:
            liked.add(row["topic"])
        elif row["avg_rating"] <= DISLIKED_RATING:
            disliked.add(row["topic"])
    for row in conn.execute("SELECT f.tags, a.topics FROM feedback f JOIN articles_seen a USING (url)"):
        tags = {t.lower() for t in json.loads(row["tags"] or "[]")}
        topics = json.loads(row["topics"] or "[]")
        if "more" in tags:
            liked.update(topics)
        if "less" in tags:
            disliked.update(topics)

    under_covered = [
        interest for interest, _ in sorted(interests.items(), key=lambda kv: -kv[1])
        if not any(interest_matches_topic(interest, topic) for topic in recent)
    ]
    return {
        "recent_topics": dict(recent.most_common()),
        "liked_topics": sorted(liked - disliked),
        "disliked_topics": sorted(disliked),
        "under_covered": under_covered,
    }


# --- Writes ------------------------------------------------------------------

def record_brief(conn: sqlite3.Connection, run_date: date, selected: list[Article]) -> None:
    """Insert the selected articles into articles_seen and bump topic_progress exposures."""
    with conn:
        for a in selected:
            published = a.get("published")
            conn.execute(
                "INSERT OR IGNORE INTO articles_seen (url, title, source, published, topics, score, shown_on) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (a["url"], a["title"], a["source"], published.isoformat() if published else None,
                 json.dumps(a.get("topics", [])), a.get("total_score"), run_date.isoformat()),
            )
            for topic in a.get("topics", []):
                conn.execute(
                    "INSERT INTO topic_progress (topic, exposures, last_seen) VALUES (?, 1, ?) "
                    "ON CONFLICT(topic) DO UPDATE SET exposures = exposures + 1, last_seen = excluded.last_seen",
                    (topic, run_date.isoformat()),
                )


def record_feedback(conn: sqlite3.Connection, run_date: date, feedback: list[dict]) -> None:
    """Store ratings/tags and update topic_progress.avg_rating for the rated articles' topics."""
    if not feedback:
        return
    with conn:
        for item in feedback:
            conn.execute(
                "INSERT INTO feedback (url, rating, tags, given_on) VALUES (?, ?, ?, ?)",
                (item["url"], item.get("rating"), json.dumps(item.get("tags", [])), run_date.isoformat()),
            )
        ratings: dict[str, list[int]] = {}
        for row in conn.execute("SELECT f.rating, a.topics FROM feedback f JOIN articles_seen a USING (url) "
                                "WHERE f.rating IS NOT NULL"):
            for topic in json.loads(row["topics"] or "[]"):
                ratings.setdefault(topic, []).append(row["rating"])
        for topic, values in ratings.items():
            conn.execute(
                "INSERT INTO topic_progress (topic, exposures, avg_rating) VALUES (?, 0, ?) "
                "ON CONFLICT(topic) DO UPDATE SET avg_rating = excluded.avg_rating",
                (topic, sum(values) / len(values)),
            )


# --- Vector store (novelty / near-duplicates) --------------------------------

def load_vector_store(embeddings, path=None):
    """InMemoryVectorStore loaded from `path` (empty store if the file doesn't exist)."""
    from langchain_core.vectorstores import InMemoryVectorStore

    path = path or config.VECTOR_STORE_PATH
    if path.exists():
        return InMemoryVectorStore.load(str(path), embeddings)
    return InMemoryVectorStore(embeddings)


def max_similarities(store, texts: list[str]) -> list[float]:
    """For each text, the highest cosine similarity to any past summary (0.0 if the store is empty).

    Embeds all texts in one call.
    """
    if not texts or not store.store:
        return [0.0] * len(texts)
    vectors = store.embeddings.embed_documents(texts)
    return [store.similarity_search_with_score_by_vector(v, k=1)[0][1] for v in vectors]


def add_summaries(store, articles: list[Article], summaries: list[ArticleSummary],
                  path=None) -> None:
    """Add today's articles (title + summary, metadata: url) and persist the store."""
    if not summaries:
        return
    by_url = {a["url"]: a for a in articles}
    store.add_texts(
        [f"{by_url[s['url']]['title']}\n{s['summary']}" for s in summaries],
        metadatas=[{"url": s["url"]} for s in summaries],
    )
    config.DATA_DIR.mkdir(exist_ok=True)
    store.dump(str(path or config.VECTOR_STORE_PATH))
