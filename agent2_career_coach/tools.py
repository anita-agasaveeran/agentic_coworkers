"""External tools: opportunity and resource search, posting checks, and the code runner.

Page loading, link checks, and arXiv search reuse Agent 1's tested implementations
(agent1_daily_tech_upgrade/tools.py). Nodes call these as plain functions; AGENT_TOOLS
exposes them as LangChain tools in case a node is later turned into a tool-calling agent.
Every function returns metadata taken from the source response, never from an LLM.
"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from langchain_core.tools import tool

import config
from state import Opportunity, OpportunityType

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CareerCoachAgent/0.1; student project)",
           "Accept": "application/json"}
SNIPPET_CHARS = 800
MAX_TEST_OUTPUT_CHARS = 3000


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

def _time_range(days: int) -> str:
    """Tavily general search has no day count; use the smallest range that covers `days`."""
    return "week" if days <= 7 else "month" if days <= 31 else "year"


def canonical_url(url: str) -> str:
    """https, no tracking parameters or fragment (job boards return both http and https)."""
    url = agent1_tools().clean_url(url)
    return "https://" + url.removeprefix("http://") if url.startswith("http://") else url


def is_posting_url(url: str, source: dict) -> bool:
    """False for search/listing pages and, for job boards, URLs that aren't a single posting."""
    if agent1_tools().is_listing_url(url):
        return False
    pattern = source.get("posting_path")
    return not pattern or re.search(pattern, urlparse(url).path) is not None


def posting_title(title: str) -> tuple[str, str | None]:
    """Job-board page title -> (role title, organization if the title names it).

    "Job Application for ML Intern (Summer 2027) at Klaviyo" -> ("ML Intern (Summer 2027)", "Klaviyo")
    """
    title = " ".join(title.split())
    title = re.sub(r"\s+[-|–]\s+Myworkdayjobs\.com$", "", title)
    match = re.fullmatch(r"Job Application for (.+) at (.+)", title)
    return (match[1], match[2]) if match else (title, None)


def _tavily_search(query: str, sources: list[dict], **kwargs) -> list[tuple[dict, dict]]:
    """(result, source) pairs from a Tavily search restricted to the sources' domains."""
    from tavily import TavilyClient   # imported here so the rest of the module works without a key

    domains = [d for s in sources for d in s["domains"]]
    response = TavilyClient().search(query, topic="general", include_domains=domains, **kwargs)
    pairs = []
    for result in response.get("results", []):
        source = agent1_tools().source_for_url(result["url"], sources)
        if source is not None:
            pairs.append((result, source))
    return pairs


def search_opportunities(query: str, opportunity_type: OpportunityType, sources: list[dict],
                         days: int, max_results: int = 8) -> list[Opportunity]:
    """Domain-restricted Tavily search for individual postings of one opportunity type.

    Searches only sources listed for `opportunity_type`. Tavily doesn't date job-board pages,
    so `days` sets its time range (week/month/year) rather than a filter on the results.
    Title, organization, and URL come from the search result.
    """
    sources = [s for s in sources if opportunity_type in s["types"]]
    if not sources:
        return []
    candidates, seen = [], set()
    for result, source in _tavily_search(query, sources, time_range=_time_range(days),
                                         max_results=max_results):
        url = canonical_url(result["url"])
        if url in seen or not is_posting_url(url, source):
            continue
        seen.add(url)
        title, org = posting_title(result.get("title", ""))
        candidates.append(Opportunity(
            url=url,
            title=title,
            org=org,
            type=opportunity_type,
            source=source["name"],
            snippet=" ".join(result.get("content", "").split())[:SNIPPET_CHARS],
        ))
    return candidates


def search_resources(query: str, sources: list[dict], max_results: int = 5) -> list[dict]:
    """Learning resources (docs, courses, tutorials) from allow-listed domains via Tavily.

    Returns [{"url", "title", "source", "snippet"}]; callers link-check with url_resolves.
    """
    resources, seen = [], set()
    for result, source in _tavily_search(query, sources, max_results=max_results):
        url = canonical_url(result["url"])
        if url in seen or agent1_tools().is_listing_url(url):
            continue
        seen.add(url)
        resources.append({
            "url": url,
            "title": " ".join(result.get("title", "").split()),
            "source": source["name"],
            "snippet": " ".join(result.get("content", "").split())[:SNIPPET_CHARS],
        })
    return resources


WORKDAY_URL = re.compile(r"https://(?P<host>(?P<tenant>[^./]+)\.wd\d+\.myworkdayjobs\.com)/"
                         r"(?:[a-z]{2}-[A-Z]{2}/)?(?P<site>[^/]+)/job/(?P<path>.+)")


def _load_workday(match: re.Match, timeout: float) -> dict | None:
    """Workday posting from its JSON endpoint (the HTML page is rendered by JavaScript)."""
    api = f"https://{match['host']}/wday/cxs/{match['tenant']}/{match['site']}/job/{match['path']}"
    try:
        response = requests.get(api, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        info = response.json().get("jobPostingInfo", {})
    except (requests.RequestException, ValueError):
        return None
    description = agent1_tools().extract_main_text(info.get("jobDescription", ""))
    if not description:
        return None
    text = "\n".join(line for line in (info.get("title"), info.get("location"), description) if line)
    start = info.get("startDate")
    return {"text": text, "word_count": len(text.split()),
            "published": date.fromisoformat(start) if start else None,
            "can_apply": info.get("canApply")}


def load_posting(url: str, timeout: float = 20) -> dict | None:
    """Posting text as {"text", "word_count", "published", "can_apply"}, or None if it can't be read.

    Workday postings are read from Workday's JSON endpoint, which also says whether the
    posting still accepts applications; other pages go through load_page (can_apply None).
    """
    if match := WORKDAY_URL.match(url):
        if posting := _load_workday(match, timeout):
            return posting
    page = load_page(url, timeout)
    return {**page, "can_apply": None} if page else None


CLOSED_PATTERNS = re.compile("|".join([
    r"no longer accepting applications",
    r"(?:job|position|posting|role|opportunity) (?:is|has been) (?:no longer available|closed|filled|expired)",
    r"(?:job|position|role) has been filled",
    r"(?:job|posting) has expired",
    r"applications? (?:are|is|has) (?:now )?closed",
    r"application (?:period|window|deadline) has (?:passed|closed|ended)",
    r"job not found",
    r"couldn['’]t find anything here",                     # Lever, removed posting
    r"the page you are looking for doesn['’]t exist",      # Workday, removed posting
]), re.IGNORECASE)


def posting_is_open(posting: dict, deadline: date | None, today: date) -> bool:
    """False if the deadline has passed, the job board says it's closed, or the text says so.

    `posting` is a load_posting result.
    """
    if deadline is not None and deadline < today:
        return False
    if posting.get("can_apply") is False:
        return False
    return CLOSED_PATTERNS.search(posting.get("text") or "") is None


def run_tests(code: str, tests: str, timeout: float = config.CODE_TIMEOUT_SECONDS) -> dict:
    """Run pytest `tests` (which import from `solution`) against `code` in a temporary directory.

    Returns {"passed": n, "total": n, "pass_rate": 0-1, "results": {test name: bool},
             "output": "<trimmed pytest output>", "timed_out": bool}.
    The subprocess gets a minimal environment (no API keys) and a time limit. It runs
    my own answers and LLM-written tests on my machine; this is not a sandbox.
    """
    with tempfile.TemporaryDirectory(prefix="coach-tests-") as tmp:
        tmp = Path(tmp)
        (tmp / "solution.py").write_text(code)
        (tmp / "test_solution.py").write_text(tests)
        report = tmp / "report.xml"
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--rootdir", str(tmp), f"--junitxml={report}", "test_solution.py"]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(command, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": 0, "pass_rate": 0.0, "results": {},
                    "output": f"timed out after {timeout:g}s", "timed_out": True}
        results = _junit_results(report) if report.exists() else {}
    passed = sum(results.values())
    output = (proc.stdout + proc.stderr).strip()
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
        "output": output[-MAX_TEST_OUTPUT_CHARS:],
        "timed_out": False,
    }


def _junit_results(report: Path) -> dict[str, bool]:
    """Test name -> passed, from pytest's JUnit XML (skipped tests count as not passed)."""
    return {case.get("name"): not any(child.tag in ("failure", "error", "skipped") for child in case)
            for case in ElementTree.parse(report).iter("testcase")}


AGENT_TOOLS = [tool(f) for f in (search_opportunities, search_resources, search_arxiv, load_page, load_posting)]
