"""Graph nodes for DailyTechUpgradeAgent (one function per node in ARCHITECTURE.md §4).

Each node takes the current BriefState and returns a partial state update.
"""

from typing import Literal

from langgraph.types import interrupt

from config import MAX_RETRIES, TOP_K
from state import BriefState


def load_profile(state: BriefState) -> dict:
    """Load profile.json and add learning-progress signals from history.db.

    Returns: {"profile": {...profile.json, "progress": memory.learning_progress(...)},
              "retries": 0, "lookback_days": config.LOOKBACK_DAYS}
    """
    raise NotImplementedError


def plan_queries(state: BriefState) -> dict:
    """LLM (QueryPlan) turns the profile and progress into search queries.

    Returns: {"queries": [...]}
    """
    raise NotImplementedError


def fetch_candidates(state: BriefState) -> dict:
    """Collect candidates from feeds, arXiv, and Tavily for each query within `lookback_days`.

    Returns: {"candidates": [...]}. A failing tool is logged and skipped, not fatal.
    """
    raise NotImplementedError


def dedupe_and_filter(state: BriefState) -> dict:
    """Normalize URLs; drop seen, near-duplicate, non-allow-listed, undated, and out-of-window items.

    Returns: {"filtered": [...]}
    """
    raise NotImplementedError


def score_and_rank(state: BriefState) -> dict:
    """LLM (CandidateScores) rates relevance/depth/novelty; code adds authority and recency.

    Weighted total with config.SCORE_WEIGHTS, then pick up to TOP_K with total >= SCORE_THRESHOLD,
    at most one per source and distinct primary topics.
    Returns: {"filtered": [...with scores], "selected": [...]}
    """
    raise NotImplementedError


def route_after_scoring(state: BriefState) -> Literal["broaden_queries", "extract_full_text"]:
    """Retry with broader queries until TOP_K articles pass, at most MAX_RETRIES times."""
    if len(state.get("selected", [])) < TOP_K and state.get("retries", 0) < MAX_RETRIES:
        return "broaden_queries"
    return "extract_full_text"


def broaden_queries(state: BriefState) -> dict:
    """LLM (QueryPlan) rewrites queries; widen the window to BROADENED_LOOKBACK_DAYS.

    Returns: {"queries": [...], "lookback_days": ..., "retries": retries + 1}
    """
    raise NotImplementedError


def extract_full_text(state: BriefState) -> dict:
    """Link-check and fetch each selected article; fall back to the snippet when extraction fails.

    Unreachable URLs are replaced by the next-ranked candidate from `filtered`.
    Returns: {"selected": [...with full_text, word_count, text_source]}
    """
    raise NotImplementedError


def summarize_and_personalize(state: BriefState) -> dict:
    """LLM (SummaryOutput) per article; read_minutes computed in code from the time budget.

    Returns: {"summaries": [ArticleSummary, ...]} in the same order as `selected`.
    """
    raise NotImplementedError


def compose_brief(state: BriefState) -> dict:
    """Render the Markdown brief (format in ARCHITECTURE.md §8) and write briefs/YYYY-MM-DD.md.

    Title, source, date, and URL come from `selected`, never from summaries.
    Returns: {"brief_markdown": ..., "brief_path": ...}
    """
    raise NotImplementedError


def collect_feedback(state: BriefState) -> dict:
    """Pause for 1-5 ratings and more/less tags per article (human-in-the-loop)."""
    if not state.get("interactive", True):
        return {"feedback": []}
    feedback = interrupt({
        "brief_path": state.get("brief_path"),
        "articles": [{"url": a["url"], "title": a["title"]} for a in state.get("selected", [])],
    })
    return {"feedback": feedback or []}


def update_memory(state: BriefState) -> dict:
    """Write selected articles, topics, and feedback to history.db; add summaries to the vector store."""
    raise NotImplementedError
