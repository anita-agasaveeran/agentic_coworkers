"""Graph state and structured-output schemas for DailyTechUpgradeAgent."""

from datetime import date
from typing import TypedDict

from pydantic import BaseModel, Field


class Article(TypedDict, total=False):
    url: str
    title: str
    source: str              # display name from sources.yaml, e.g. "AWS Machine Learning Blog"
    tier: int                # authority tier from sources.yaml
    published: date
    snippet: str             # from the feed / search result
    full_text: str | None
    word_count: int | None
    topics: list[str]
    scores: dict[str, float]     # relevance, depth, novelty, authority, recency (each 0-1)
    total_score: float
    text_source: str         # "full_text" or "snippet" (set by extract_full_text)


class ArticleSummary(TypedDict):
    url: str                 # joins back to Article; metadata is never taken from the LLM
    summary: str
    key_takeaways: list[str]
    why_it_matters: str
    think_about: str
    reading_scope: str       # e.g. "Read §3 Method and §5 Results"
    read_minutes: int


class BriefState(TypedDict, total=False):
    run_date: date
    profile: dict
    queries: list[str]
    lookback_days: int
    candidates: list[Article]
    filtered: list[Article]
    selected: list[Article]
    summaries: list[ArticleSummary]
    brief_markdown: str
    brief_path: str
    feedback: list[dict]     # {"url", "rating" (1-5), "tags": ["more" | "less" | ...]}
    retries: int
    interactive: bool        # False skips the collect_feedback interrupt


# --- Structured LLM outputs --------------------------------------------------

class QueryPlan(BaseModel):
    queries: list[str] = Field(description="Search queries for today's candidate articles")


class CandidateScore(BaseModel):
    url: str
    relevance: int = Field(ge=1, le=5, description="Fit with the reader's interests and goals")
    depth: int = Field(ge=1, le=5, description="Technical depth; 1 = announcement, 5 = deep technical content")
    novelty: int = Field(ge=1, le=5, description="New to this reader given their history")
    topics: list[str] = Field(description="1-3 short topic labels")


class CandidateScores(BaseModel):
    scores: list[CandidateScore]


class SummaryOutput(BaseModel):
    summary: str = Field(description="4-6 sentence technical summary")
    key_takeaways: list[str] = Field(min_length=3, max_length=5)
    why_it_matters: str = Field(description="How this helps the reader's specific goals")
    think_about: str = Field(description="One question to reflect on after reading")
    reading_scope: str = Field(description="What to read to fit the time budget")
