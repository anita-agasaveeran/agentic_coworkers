"""External tools: opportunity and resource search, posting checks, and the code runner.

Page loading, link checks, and arXiv search reuse Agent 1's tested implementations
(agent1_daily_tech_upgrade/tools.py). Nodes call these as plain functions; AGENT_TOOLS
exposes them as LangChain tools in case a node is later turned into a tool-calling agent.
Every function returns metadata taken from the source response, never from an LLM.
"""

import importlib.util
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

import config


# --- Agent 1 tools -----------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache
def agent1_tools():
    """Agent 1's tools module, loaded as `agent1_tools`.

    Both agents have a top-level `state` module, so Agent 1's is put in sys.modules
    only while its tools.py runs, then ours is restored.
    """
    ours = sys.modules.get("state")
    sys.modules["state"] = _load_module("agent1_state", config.AGENT1_DIR / "state.py")
    try:
        return _load_module("agent1_tools", config.AGENT1_DIR / "tools.py")
    finally:
        if ours is None:
            sys.modules.pop("state")
        else:
            sys.modules["state"] = ours


def load_page(url: str, timeout: float = 20) -> dict | None:
    """Fetch a page and return {"text", "word_count", "published"}, or None if it can't be read."""
    return agent1_tools().load_page(url, timeout)


def url_resolves(url: str, timeout: float = 10) -> bool:
    """True unless the link is broken (404/410, 5xx, or no connection); 401/403/429 count as resolving."""
    return agent1_tools().url_resolves(url, timeout)


def search_arxiv(query: str, categories: list[str], since: date, max_results: int = 10) -> list[dict]:
    """Recent arXiv papers matching `query` in the given categories (newest first)."""
    return agent1_tools().search_arxiv(query, categories, since, max_results)


# --- Agent 2 tools -----------------------------------------------------------

def search_opportunities(query: str, sources: list[dict], days: int, max_results: int = 8) -> list[dict]:
    """Domain-restricted Tavily search for individual postings; returns Opportunity candidates."""
    # TODO: TavilyClient().search(query, topic="general", include_domains=<sources' domains>,
    #       max_results=max_results), keeping results from the last `days` days when dated.
    #       Map each result -> {"url", "title", "snippet", "source", "type" (from the query)};
    #       drop listing/search pages (reuse agent1_tools().is_listing_url) and clean URLs.
    raise NotImplementedError


def search_resources(query: str, sources: list[dict], max_results: int = 5) -> list[dict]:
    """Learning resources (docs, courses, tutorials) from allow-listed domains via Tavily."""
    # TODO: TavilyClient().search(query, topic="general", include_domains=...); map to
    #       {"url", "title", "source", "snippet"}. Callers link-check with url_resolves.
    raise NotImplementedError


def posting_is_open(page_text: str, deadline: date | None, today: date) -> bool:
    """False if the deadline has passed or the page says applications are closed."""
    # TODO: deadline < today -> False; regex for phrases like "no longer accepting applications",
    #       "this job is no longer available", "position has been filled", "applications are closed".
    raise NotImplementedError


def run_tests(code_path: Path, tests: str, timeout: float = config.CODE_TIMEOUT_SECONDS) -> dict:
    """Run pytest `tests` against my solution in a temporary directory.

    Returns {"passed": n, "total": n, "pass_rate": 0-1, "output": "<trimmed pytest output>",
             "timed_out": bool}. Only my own answers are run; this is not a sandbox.
    """
    # TODO: copy code_path -> tmp/solution.py, write tmp/test_solution.py, run
    #       [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", tmp] with
    #       subprocess.run(timeout=timeout, capture_output=True, cwd=tmp); parse the
    #       "N passed, M failed" summary line. Timeout -> pass_rate 0.
    raise NotImplementedError


AGENT_TOOLS = [tool(f) for f in (search_opportunities, search_resources, search_arxiv, load_page)]
