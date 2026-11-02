"""Graph nodes for DailyTechUpgradeAgent (one function per node in ARCHITECTURE.md §4).

Each node takes the current BriefState and returns a partial state update.
"""

import json
import logging
import math
import os
import re
from datetime import date, timedelta
from typing import Literal
from urllib.parse import urlparse, urlunparse

from langgraph.types import interrupt

import config
import memory
import prompts
import tools
from state import Article, ArticleSummary, BriefState, CandidateScores, QueryPlan, SummaryOutput

log = logging.getLogger(__name__)


# --- Helpers -----------------------------------------------------------------

def structured(schema):
    """The chat model constrained to `schema` via native structured outputs (no forced tool use)."""
    return config.get_llm().with_structured_output(schema, method="json_schema")


def embeddings_or_none():
    """Embeddings model, or None when it isn't configured (novelty checks are then skipped)."""
    try:
        return config.get_embeddings()
    except Exception as exc:   # missing key or provider package
        log.warning("embeddings unavailable, skipping near-duplicate checks (%s)", exc)
        return None


def profile_for_prompt(profile: dict) -> str:
    return json.dumps({k: v for k, v in profile.items() if not k.startswith("_") and k != "progress"}, indent=2)


def normalize_url(url: str) -> str:
    """Canonical form for de-duplication: lowercase host, no www, no trailing slash, no arXiv version."""
    parts = urlparse(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    if host == "arxiv.org":
        path = re.sub(r"v\d+$", "", path.replace("/pdf/", "/abs/"))
    return urlunparse(("https", host, path, "", parts.query, ""))


def clamp_score(value: int) -> float:
    """1-5 integer score -> 0-1."""
    return (min(max(int(value), 1), 5) - 1) / 4


def recency_score(published: date | None, run_date: date, lookback_days: int) -> float:
    if published is None:
        return 0.5
    age = (run_date - published).days
    return min(max(1 - age / max(lookback_days, 1), 0.0), 1.0)


def is_diverse(article: Article, picked: list[Article]) -> bool:
    """At most one article per source and distinct primary topics."""
    topic = (article.get("topics") or [""])[0].lower()
    return all(p["source"] != article["source"] and (p.get("topics") or [""])[0].lower() != topic for p in picked)


def matched_interest(article: Article, interests: dict[str, float]) -> str | None:
    """The profile interest an article serves: the scorer's label, else the best keyword match."""
    found = memory.article_interests(article, interests)
    return found[0] if found else None


def pick_diverse(ranked: list[Article], k: int, threshold: float,
                 interests: dict[str, float] | None = None) -> list[Article]:
    """Top-k passing articles with distinct sources and primary topics.

    Each step takes the best remaining article, unless one for a not-yet-covered profile
    interest scores within INTEREST_SPREAD_MARGIN of it; then that one is taken instead,
    so the brief spreads across interests without picking much weaker articles.
    """
    remaining = [a for a in ranked if a.get("total_score", 0) >= threshold]
    picked: list[Article] = []
    covered: set[str] = set()
    while len(picked) < k:
        options = [a for a in remaining if is_diverse(a, picked)]
        if not options:
            break
        best = options[0]
        fresh = next((a for a in options
                      if (i := matched_interest(a, interests or {})) and i not in covered), None)
        if fresh and fresh["total_score"] >= best["total_score"] - config.INTEREST_SPREAD_MARGIN:
            best = fresh
        picked.append(best)
        remaining.remove(best)
        if interest := matched_interest(best, interests or {}):
            covered.add(interest)
    return sorted(picked, key=lambda a: a.get("total_score", 0), reverse=True)


def slot_minutes(n_articles: int) -> int:
    """Reading minutes available per article after the brief itself."""
    return max((config.SESSION_MINUTES[1] - config.BRIEF_MINUTES) // max(n_articles, 1), 1)


# --- Nodes -------------------------------------------------------------------

def load_profile(state: BriefState) -> dict:
    """Load profile.json and add learning-progress signals from history.db."""
    run_date = state.get("run_date") or date.today()
    profile = config.load_profile_file()
    conn = memory.connect()
    try:
        progress = memory.learning_progress(conn, run_date, profile.get("interests", {}))
    finally:
        conn.close()
    return {
        "run_date": run_date,
        "profile": {**profile, "progress": progress},
        "retries": 0,
        "lookback_days": config.LOOKBACK_DAYS,
    }


def plan_queries(state: BriefState) -> dict:
    """LLM turns the profile and progress into search queries."""
    lo, hi = config.NUM_QUERIES
    plan = structured(QueryPlan).invoke(prompts.PLAN_QUERIES.format(
        profile=profile_for_prompt(state["profile"]),
        progress=json.dumps(state["profile"]["progress"], indent=2),
        min_queries=lo, max_queries=hi, arxiv_queries=config.ARXIV_QUERIES,
    ))
    return queries_update(plan)


def queries_update(plan: QueryPlan) -> dict:
    return {
        "queries": [q.strip() for q in plan.queries if q.strip()][:config.NUM_QUERIES[1]],
        "arxiv_queries": [q.strip() for q in plan.arxiv_queries if q.strip()][:config.ARXIV_QUERIES],
    }


def fetch_candidates(state: BriefState) -> dict:
    """Collect candidates from feeds, Tavily (each web query), and arXiv (each arXiv query) within the window."""
    since = state["run_date"] - timedelta(days=state["lookback_days"])
    sources_cfg = config.load_sources()
    sources = sources_cfg["sources"]

    candidates = tools.fetch_feeds([s for s in sources if s.get("feeds")], since)

    if os.getenv("TAVILY_API_KEY"):
        for query in state["queries"]:
            try:
                candidates += tools.search_web(query, sources, state["lookback_days"], config.MAX_RESULTS_PER_QUERY)
            except Exception as exc:
                log.warning("web search failed: %s (%s)", query, exc)
    else:
        log.warning("TAVILY_API_KEY not set; skipping web search (sources without feeds are not covered)")

    arxiv = sources_cfg.get("arxiv")
    if arxiv:
        for query in state.get("arxiv_queries", []):
            candidates += tools.search_arxiv(query, arxiv["categories"], since,
                                             arxiv["max_results_per_query"], arxiv["tier"])

    unique: dict[str, Article] = {}
    for article in candidates:
        key = normalize_url(article["url"])
        if key not in unique or (unique[key].get("published") is None and article.get("published")):
            unique[key] = article
    log.info("fetched %d candidates (%d unique)", len(candidates), len(unique))
    return {"candidates": list(unique.values())}


def dedupe_and_filter(state: BriefState) -> dict:
    """Drop seen, out-of-window, duplicate-title, and near-duplicate (vs. past summaries) items.

    Undated search results are kept (Tavily's `days` window already bounds them); their date is
    filled in by extract_full_text. Allow-listing happens in the tools (feeds come from
    sources.yaml; search results outside its domains are dropped).
    """
    run_date = state["run_date"]
    since = run_date - timedelta(days=state["lookback_days"])
    conn = memory.connect()
    try:
        seen = {normalize_url(u) for u in memory.seen_urls(conn)}
    finally:
        conn.close()

    kept, titles = [], set()
    for article in state["candidates"]:
        published = article.get("published")
        title_key = re.sub(r"\W+", " ", article["title"].lower()).strip()
        if normalize_url(article["url"]) in seen or title_key in titles:
            continue
        if published is not None and not (since <= published <= run_date):
            continue
        titles.add(title_key)
        kept.append(article)

    embeddings = embeddings_or_none()
    if embeddings is not None and kept:
        store = memory.load_vector_store(embeddings)
        sims = memory.max_similarities(store, [f"{a['title']}\n{a['snippet']}" for a in kept])
        kept = [a for a, sim in zip(kept, sims) if sim < config.NEAR_DUPLICATE_SIMILARITY]

    log.info("kept %d of %d candidates", len(kept), len(state["candidates"]))
    return {"filtered": kept}


def score_and_rank(state: BriefState) -> dict:
    """LLM rates relevance/depth/novelty; code adds authority and recency, ranks, and picks TOP_K."""
    profile = state["profile"]
    candidates = state["filtered"]
    scorer = structured(CandidateScores)
    llm_scores: dict[int, object] = {}
    for start in range(0, len(candidates), config.SCORE_BATCH_SIZE):
        batch = candidates[start:start + config.SCORE_BATCH_SIZE]
        listing = "\n".join(
            f"[{start + i}] {a['title']} | {a['source']} | {a['snippet'][:400]}" for i, a in enumerate(batch))
        result = scorer.invoke(prompts.SCORE_CANDIDATES.format(
            profile=profile_for_prompt(profile),
            recent_topics=json.dumps(profile["progress"]["recent_topics"]),
            liked_topics=", ".join(profile["progress"]["liked_topics"]) or "none yet",
            disliked_topics=", ".join(profile["progress"]["disliked_topics"]) or "none yet",
            candidates=listing,
        ))
        llm_scores.update({s.id: s for s in result.scores if start <= s.id < start + len(batch)})

    weights = config.SCORE_WEIGHTS
    interest_keys = {k.lower(): k for k in profile.get("interests", {})}
    scored: list[Article] = []
    for i, article in enumerate(candidates):
        s = llm_scores.get(i)
        if s is None:
            continue
        scores = {
            "relevance": clamp_score(s.relevance),
            "depth": clamp_score(s.depth),
            "novelty": clamp_score(s.novelty),
            "authority": config.TIER_AUTHORITY.get(article.get("tier"), 0.5),
            "recency": recency_score(article.get("published"), state["run_date"], state["lookback_days"]),
        }
        scored.append({
            **article,
            "topics": [t.strip().lower() for t in s.topics if t.strip()][:3],
            "interest": interest_keys.get(s.interest.strip().lower()),
            "scores": scores,
            "total_score": round(sum(weights[k] * scores[k] for k in weights), 4),
        })

    scored.sort(key=lambda a: a["total_score"], reverse=True)
    selected = pick_diverse(scored, config.TOP_K, config.SCORE_THRESHOLD, profile.get("interests"))
    log.info("scored %d candidates, selected %d", len(scored), len(selected))
    return {"filtered": scored, "selected": selected}


def route_after_scoring(state: BriefState) -> Literal["broaden_queries", "extract_full_text"]:
    """Retry with broader queries until TOP_K articles pass, at most MAX_RETRIES times."""
    if len(state.get("selected", [])) < config.TOP_K and state.get("retries", 0) < config.MAX_RETRIES:
        return "broaden_queries"
    return "extract_full_text"


def broaden_queries(state: BriefState) -> dict:
    """LLM rewrites queries; widen the window to BROADENED_LOOKBACK_DAYS."""
    lo, hi = config.NUM_QUERIES
    plan = structured(QueryPlan).invoke(prompts.BROADEN_QUERIES.format(
        num_passing=len(state.get("selected", [])),
        queries="\n".join(state["queries"] + state.get("arxiv_queries", [])),
        profile=profile_for_prompt(state["profile"]),
        min_queries=lo, max_queries=hi, arxiv_queries=config.ARXIV_QUERIES,
    ))
    return {
        **queries_update(plan),
        "lookback_days": config.BROADENED_LOOKBACK_DAYS,
        "retries": state.get("retries", 0) + 1,
    }


def extract_full_text(state: BriefState) -> dict:
    """Link-check and fetch each selected article.

    Broken links are replaced by the next-ranked passing candidate that keeps the picks diverse.
    Articles whose text can't be retrieved are used (summarized from the snippet) only when no
    full-text candidate is left to fill the slot. Fills `published` from the page when the
    search result had no date; still undated -> replaced.
    """
    target = len(state.get("selected", []))
    selected_urls = {a["url"] for a in state.get("selected", [])}
    backups = [a for a in state.get("filtered", [])
               if a["url"] not in selected_urls and a.get("total_score", 0) >= config.SCORE_THRESHOLD]

    final: list[Article] = []
    snippet_only: list[Article] = []
    for article in state.get("selected", []) + backups:
        if len(final) == target:
            break
        if not is_diverse(article, final) or not tools.url_resolves(article["url"]):
            continue
        page = tools.load_page(article["url"])
        published = article.get("published") or (page or {}).get("published")
        if published is None:
            continue
        if page is None:
            snippet_only.append({**article, "published": published, "full_text": None,
                                 "word_count": None, "text_source": "snippet"})
            continue
        final.append({**article, "published": published, "full_text": page["text"],
                      "word_count": page["word_count"], "text_source": "full_text"})
    for article in snippet_only:
        if len(final) < target and is_diverse(article, final):
            final.append(article)
    final.sort(key=lambda a: a.get("total_score", 0), reverse=True)
    return {"selected": final}


def summarize_and_personalize(state: BriefState) -> dict:
    """LLM summary per article; read_minutes computed in code from the time budget."""
    selected = state.get("selected", [])
    if not selected:
        return {"summaries": []}
    slot = slot_minutes(len(selected))
    requests_ = []
    for a in selected:
        text = a.get("full_text") or a["snippet"]
        if len(text) > config.MAX_TEXT_CHARS:
            log.warning("truncating %s from %d to %d chars", a["url"], len(text), config.MAX_TEXT_CHARS)
            text = text[:config.MAX_TEXT_CHARS]
        requests_.append(prompts.SUMMARIZE_ARTICLE.format(
            minutes=slot,
            profile=profile_for_prompt(state["profile"]),
            title=a["title"], source=a["source"], published=a["published"].isoformat(),
            text_source="full article" if a.get("full_text") else "feed snippet only",
            text=text,
        ))
    outputs: list[SummaryOutput] = structured(SummaryOutput).batch(requests_)

    summaries: list[ArticleSummary] = []
    for a, out in zip(selected, outputs):
        words = a.get("word_count")
        read_minutes = min(math.ceil(words / config.WORDS_PER_MINUTE), slot) if words else slot
        summaries.append(ArticleSummary(
            url=a["url"],
            summary=out.summary.strip(),
            key_takeaways=[t.strip() for t in out.key_takeaways if t.strip()][:5],
            why_it_matters=out.why_it_matters.strip(),
            think_about=out.think_about.strip(),
            reading_scope=out.reading_scope.strip(),
            read_minutes=max(read_minutes, 1),
        ))
    return {"summaries": summaries}


def render_brief(run_date: date, selected: list[Article], summaries: list[ArticleSummary]) -> str:
    """Markdown brief (ARCHITECTURE.md §8). Title, source, date, and URL come from `selected` only."""
    by_url = {s["url"]: s for s in summaries}
    total = config.BRIEF_MINUTES + sum(s["read_minutes"] for s in summaries)
    day = f"{run_date:%b} {run_date.day}, {run_date.year}"
    lines = [f"# Daily Tech Brief — {day}   (≈{total} min)", ""]
    focus = [a["topics"][0] for a in selected if a.get("topics")]
    if focus:
        lines += [f"Focus today: {' · '.join(focus)}", ""]
    if not selected:
        lines += ["No articles met today's quality bar. Try again tomorrow or widen `profile.json` interests.", ""]
    elif len(selected) < config.TOP_K:
        lines += [f"_Only {len(selected)} article(s) met today's quality bar._", ""]

    for i, a in enumerate(selected, 1):
        s = by_url[a["url"]]
        lines += [
            f"## {i}. {a['title']}",
            f"Source: {a['source']} · Published: {a['published']:%b} {a['published'].day}, {a['published'].year}"
            f" · <{a['url']}> · ~{s['read_minutes']} min",
            "",
            f"**Summary:** {s['summary']}",
            "",
            "**Key takeaways:**",
            *[f"- {t}" for t in s["key_takeaways"]],
            "",
            f"**Why this matters for you:** {s['why_it_matters']}",
            "",
            f"**What to read:** {s['reading_scope']}",
            "",
            f"**Think about:** {s['think_about']}",
        ]
        if a.get("text_source") == "snippet":
            lines += ["", "_Summary based on the feed snippet; the full page couldn't be retrieved._"]
        lines.append("")

    if selected:
        lines += ["### References"]
        lines += [f"{i}. {a['title']}. {a['source']}, {a['published'].isoformat()}. {a['url']}"
                  for i, a in enumerate(selected, 1)]
    return "\n".join(lines).rstrip() + "\n"


def compose_brief(state: BriefState) -> dict:
    """Render the Markdown brief and write briefs/YYYY-MM-DD.md."""
    selected = state.get("selected", [])
    summaries = state.get("summaries", [])
    brief_urls = {s["url"] for s in summaries}
    if brief_urls != {a["url"] for a in selected}:
        raise ValueError("every summary must belong to a selected article")

    markdown = render_brief(state["run_date"], selected, summaries)
    config.BRIEFS_DIR.mkdir(exist_ok=True)
    path = config.BRIEFS_DIR / f"{state['run_date'].isoformat()}.md"
    path.write_text(markdown)
    return {"brief_markdown": markdown, "brief_path": str(path)}


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
    run_date = state["run_date"]
    selected = state.get("selected", [])
    conn = memory.connect()
    try:
        memory.record_brief(conn, run_date, selected)
        memory.record_feedback(conn, run_date, state.get("feedback", []))
    finally:
        conn.close()
    embeddings = embeddings_or_none()
    if embeddings is not None:
        memory.add_summaries(memory.load_vector_store(embeddings), selected, state.get("summaries", []))
    return {}
