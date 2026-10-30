"""Offline evaluation over saved runs (ARCHITECTURE.md §10).

    python evaluate.py                  # all saved runs
    python evaluate.py --days 7         # last 7 runs
    python evaluate.py --no-llm         # skip the LLM-as-judge faithfulness check
    python evaluate.py --no-network     # skip live link/metadata checks

Writes a Markdown report to eval_reports/eval_YYYY-MM-DD.md and prints it.
"""

import argparse
import re
import sqlite3
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import requests
from bs4 import BeautifulSoup

import config
import memory
import prompts
import tools
from state import FaithfulnessJudgment

REPORTS_DIR = config.AGENT_DIR / "eval_reports"
DATE_TOLERANCE_DAYS = 2          # feed vs. page dates can differ by time zone or "updated" dates
TITLE_MATCH_RATIO = 0.8
GOOD_RATING = 4


# --- Loading -----------------------------------------------------------------

def load_runs(days: int | None = None, checkpoint_db: Path | None = None) -> list[dict]:
    """Final BriefState of each saved run (thread_id = run date), oldest first."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    import agent

    db = checkpoint_db or config.CHECKPOINT_DB
    if not db.exists():
        return []
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        threads = sorted({c.config["configurable"]["thread_id"] for c in checkpointer.list(None)})
        app = agent.build_graph(checkpointer)
        runs = [app.get_state({"configurable": {"thread_id": t}}).values for t in threads]
    runs = [r for r in runs if r.get("brief_markdown")]
    return runs[-days:] if days else runs


# --- Reference accuracy --------------------------------------------------------

def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def titles_match(stored: str, page: str) -> bool:
    a, b = _norm_title(stored), _norm_title(page)
    return bool(a and b) and (a in b or b in a or SequenceMatcher(None, a, b).ratio() >= TITLE_MATCH_RATIO)


def page_metadata(url: str) -> tuple[str | None, date | None]:
    """(title, published) as declared by the page; (None, None) if the page can't be fetched."""
    try:
        response = requests.get(url, headers=tools.HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return None, None
    soup = BeautifulSoup(response.text, "html.parser")
    og = soup.find("meta", attrs={"property": "og:title"})
    title = (og.get("content") if og else None) or (soup.title.string if soup.title else None)
    return title, tools.page_published_date(response.text)


def brief_references_consistent(run: dict) -> bool:
    """The brief's Source-line links and References list both match the selected articles exactly."""
    markdown = run["brief_markdown"]
    source_links = re.findall(r"^Source: .*?· <(https?://[^>]+)>", markdown, flags=re.M)
    references = re.findall(r"^\d+\. .*?(https?://\S+)$", markdown.split("### References")[-1], flags=re.M)
    selected = [a["url"] for a in run.get("selected", [])]
    return source_links == selected and (references == selected if selected else True)


def reference_accuracy(runs: list[dict], network: bool = True) -> dict:
    """Brief/reference consistency (offline) plus live link, title, and date checks."""
    articles = [a for r in runs for a in r.get("selected", [])]
    result = {
        "articles": len(articles),
        "briefs_with_consistent_references": f"{sum(map(brief_references_consistent, runs))}/{len(runs)}",
    }
    if not network or not articles:
        return result
    links_ok, title_checks, date_checks = 0, [], []
    for a in articles:
        links_ok += tools.url_resolves(a["url"])
        page_title, page_date = page_metadata(a["url"])
        if page_title:
            title_checks.append(titles_match(a["title"], page_title))
        if page_date:
            date_checks.append(abs((page_date - a["published"]).days) <= DATE_TOLERANCE_DAYS)
    result |= {
        "links_resolving": _ratio(links_ok, len(articles)),
        "titles_matching_page": _ratio(sum(title_checks), len(title_checks)),
        "dates_matching_page": _ratio(sum(date_checks), len(date_checks)),
        "pages_unverifiable": len(articles) - len(title_checks),
    }
    return result


# --- Summary faithfulness ------------------------------------------------------

def judge(article: dict, summary: dict) -> FaithfulnessJudgment:
    llm = config.get_llm().with_structured_output(FaithfulnessJudgment, method="json_schema")
    return llm.invoke(prompts.JUDGE_FAITHFULNESS.format(
        text_source="full article" if article.get("full_text") else "feed snippet only",
        text=(article.get("full_text") or article["snippet"])[:config.MAX_TEXT_CHARS],
        summary=summary["summary"],
        takeaways="\n".join(f"- {t}" for t in summary["key_takeaways"]),
    ))


def summary_faithfulness(runs: list[dict], judge_fn=judge) -> dict:
    """LLM-as-judge: 1-5 support score per summary, plus the unsupported claims it found."""
    scores, flagged = [], []
    for run in runs:
        by_url = {a["url"]: a for a in run.get("selected", [])}
        for summary in run.get("summaries", []):
            verdict = judge_fn(by_url[summary["url"]], summary)
            score = min(max(verdict.score, 1), 5)
            scores.append(score)
            flagged += [f"{by_url[summary['url']]['title'][:60]}: {c}" for c in verdict.unsupported_claims]
    return {
        "summaries_judged": len(scores),
        "mean_faithfulness_1_5": round(mean(scores), 2) if scores else None,
        "share_scored_4_or_5": _ratio(sum(s >= 4 for s in scores), len(scores)),
        "unsupported_claims": flagged,
    }


# --- Relevance (user feedback) -------------------------------------------------

def relevance(conn: sqlite3.Connection) -> dict:
    """Mean user rating and precision@3 (share of rated articles rated >= 4)."""
    shown = conn.execute("SELECT COUNT(*) FROM articles_seen").fetchone()[0]
    ratings = [r[0] for r in conn.execute("SELECT rating FROM feedback WHERE rating IS NOT NULL")]
    return {
        "articles_shown": shown,
        "articles_rated": len(ratings),
        "mean_rating_1_5": round(mean(ratings), 2) if ratings else None,
        "precision_at_3": _ratio(sum(r >= GOOD_RATING for r in ratings), len(ratings)),
    }


# --- Novelty, diversity, coverage ------------------------------------------------

def novelty_and_diversity(runs: list[dict], interests: dict[str, float]) -> dict:
    articles = [a for r in runs for a in r.get("selected", [])]
    urls = [a["url"] for a in articles]
    per_run_sources = [len({a["source"] for a in r["selected"]}) / len(r["selected"])
                       for r in runs if r.get("selected")]
    covered = {i for i in interests
               if any(memory.interest_matches_topic(i, t) for a in articles for t in a.get("topics", []))}
    return {
        "repeated_articles": len(urls) - len(set(urls)),
        "distinct_sources": len({a["source"] for a in articles}),
        "distinct_primary_topics": len({(a.get("topics") or [""])[0] for a in articles}),
        "mean_source_diversity_per_brief": round(mean(per_run_sources), 2) if per_run_sources else None,
        "interests_covered": f"{len(covered)}/{len(interests)}",
        "interests_not_covered": sorted(set(interests) - covered),
    }


# --- Brief quality and pipeline health -------------------------------------------

def brief_minutes(run: dict) -> int:
    return config.BRIEF_MINUTES + sum(s["read_minutes"] for s in run.get("summaries", []))


def time_budget_fit(runs: list[dict]) -> dict:
    minutes = [brief_minutes(r) for r in runs if r.get("summaries")]
    lo, hi = config.SESSION_MINUTES
    return {
        "mean_minutes": round(mean(minutes), 1) if minutes else None,
        "briefs_within_budget": _ratio(sum(lo <= m <= hi for m in minutes), len(minutes)),
    }


def pipeline_health(runs: list[dict]) -> dict:
    articles = [a for r in runs for a in r.get("selected", [])]
    return {
        "runs": len(runs),
        "mean_candidates": round(mean(len(r.get("candidates", [])) for r in runs), 1) if runs else None,
        "full_briefs": f"{sum(len(r.get('selected', [])) == config.TOP_K for r in runs)}/{len(runs)}",
        "runs_needing_retry": sum(r.get("retries", 0) > 0 for r in runs),
        "full_text_rate": _ratio(sum(a.get("text_source") == "full_text" for a in articles), len(articles)),
        "mean_selected_score": round(mean(a["total_score"] for a in articles), 3) if articles else None,
    }


# --- Learning progress -----------------------------------------------------------

def learning_progress(conn: sqlite3.Connection, runs: list[dict], interests: dict[str, float]) -> dict:
    """Articles per interest across runs, rating trend, and what the agent will steer toward next."""
    per_interest = {i: 0 for i in interests}
    for run in runs:
        for a in run.get("selected", []):
            for interest in interests:
                if any(memory.interest_matches_topic(interest, t) for t in a.get("topics", [])):
                    per_interest[interest] += 1
    by_day = conn.execute(
        "SELECT given_on, AVG(rating) FROM feedback WHERE rating IS NOT NULL GROUP BY given_on ORDER BY given_on"
    ).fetchall()
    after_last_run = max(r["run_date"] for r in runs) + timedelta(days=1) if runs else date.today()
    progress = memory.learning_progress(conn, after_last_run, interests)
    return {
        "articles_per_interest": per_interest,
        "mean_rating_by_day": {d: round(r, 2) for d, r in by_day},
        "liked_topics": progress["liked_topics"],
        "disliked_topics": progress["disliked_topics"],
        "next_focus_under_covered": progress["under_covered"],
    }


# --- Report ----------------------------------------------------------------------

def _ratio(n: int, d: int) -> str | None:
    return f"{n}/{d} ({n / d:.0%})" if d else None


def render_report(sections: dict[str, dict], runs: list[dict]) -> str:
    dates = [r["run_date"].isoformat() for r in runs]
    lines = [f"# DailyTechUpgradeAgent evaluation — {date.today().isoformat()}", "",
             f"Runs evaluated: {len(runs)}" + (f" ({dates[0]} to {dates[-1]})" if dates else ""), ""]
    for title, metrics in sections.items():
        lines += [f"## {title}", "", "| Metric | Value |", "|---|---|"]
        lists = {}
        for name, value in metrics.items():
            if isinstance(value, list) and value and name in ("unsupported_claims",):
                lists[name] = value
                value = len(value)
            if isinstance(value, dict):
                value = ", ".join(f"{k}: {v}" for k, v in value.items()) or "—"
            elif isinstance(value, list):
                value = ", ".join(map(str, value)) or "—"
            lines.append(f"| {name.replace('_', ' ')} | {'—' if value is None else value} |")
        for name, values in lists.items():
            lines += ["", f"**{name.replace('_', ' ').capitalize()}:**", *[f"- {v}" for v in values]]
        lines.append("")
    return "\n".join(lines)


def evaluate(days: int | None = None, use_llm: bool = True, network: bool = True,
             runs: list[dict] | None = None, judge_fn=judge) -> str:
    runs = load_runs(days) if runs is None else runs
    interests = config.load_profile_file().get("interests", {})
    conn = memory.connect()
    try:
        sections = {
            "Pipeline health": pipeline_health(runs),
            "Reference accuracy": reference_accuracy(runs, network),
            "Summary faithfulness (LLM-as-judge)": (summary_faithfulness(runs, judge_fn) if use_llm
                                                    else {"skipped": "--no-llm"}),
            "Relevance (your ratings)": relevance(conn),
            "Novelty and diversity": novelty_and_diversity(runs, interests),
            "Time budget": time_budget_fit(runs),
            "Learning progress": learning_progress(conn, runs, interests),
        }
    finally:
        conn.close()
    return render_report(sections, runs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DailyTechUpgradeAgent runs.")
    parser.add_argument("--days", type=int, default=None, help="only the last N runs")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM-as-judge faithfulness check")
    parser.add_argument("--no-network", action="store_true", help="skip live link and metadata checks")
    args = parser.parse_args()

    report = evaluate(args.days, use_llm=not args.no_llm, network=not args.no_network)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval_{date.today().isoformat()}.md"
    path.write_text(report)
    print(report)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
