"""External data tools: feeds, web search, arXiv, and page loading.

Nodes call these as plain functions. AGENT_TOOLS exposes the same functions
as LangChain tools in case a node is later turned into a tool-calling agent.
Every function returns metadata taken from the source response, never from an LLM.
"""

import logging
import os
import re
import time
from calendar import timegm
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from state import Article

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DailyTechUpgradeAgent/0.1; student project)"}
SNIPPET_CHARS = 800
MIN_PAGE_WORDS = 150
BOT_BLOCK_STATUSES = {401, 403, 429}
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_RETRY_SECONDS = 15
ARXIV_DELAY_SECONDS = 3          # arXiv API terms ask for at most one request every 3 seconds
ARXIV_MAX_TERMS = 4
LISTING_PARAMS = {"q", "query", "search", "s"}
ARXIV_STOPWORDS = {"and", "for", "the", "with", "from", "into", "using", "how", "what", "new"}
_last_arxiv_call = 0.0


# --- Helpers -----------------------------------------------------------------

def html_to_text(html: str) -> str:
    return " ".join(BeautifulSoup(html or "", "html.parser").get_text(" ").split())


def _entry_date(entry) -> date | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(timegm(parsed), tz=timezone.utc).date()


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace("/", "-")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).date()   # RFC 822, e.g. "Thu, 29 Oct 2026 17:00:00 GMT"
    except (TypeError, ValueError):
        return None


def clean_url(url: str) -> str:
    """Drop utm_* tracking parameters and fragments."""
    parts = urlparse(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not k.lower().startswith("utm_")])
    return urlunparse(parts._replace(query=query, fragment=""))


def source_for_url(url: str, sources: list[dict]) -> dict | None:
    """The sources.yaml entry whose domains include the URL's host (subdomains match)."""
    host = (urlparse(url).hostname or "").removeprefix("www.")
    for source in sources:
        if any(host == d or host.endswith("." + d) for d in source["domains"]):
            return source
    return None


def entries_to_articles(entries, source: dict, since: date) -> list[Article]:
    articles = []
    for entry in entries:
        published = _entry_date(entry)
        if not entry.get("link") or not published or published < since:
            continue
        articles.append(Article(
            url=clean_url(entry["link"]),
            title=" ".join(entry.get("title", "").split()),
            source=source["name"],
            tier=source["tier"],
            published=published,
            snippet=html_to_text(entry.get("summary", ""))[:SNIPPET_CHARS],
        ))
    return articles


def _word_count(tag) -> int:
    return len(tag.get_text(" ").split())


def extract_main_text(html: str) -> str:
    """Main readable text of a page, minus page chrome.

    Uses the largest <article> unless it holds less than half of <main> (pages where
    <article> tags are only related-post cards), then <main>, then <body>.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    article = max(soup.find_all("article"), key=_word_count, default=None)
    root = article if article and _word_count(article) >= 0.5 * _word_count(main) else main
    lines = (" ".join(line.split()) for line in root.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def page_published_date(html: str) -> date | None:
    """Publication date from common meta tags, if the page declares one."""
    soup = BeautifulSoup(html, "html.parser")
    for attrs in (
        {"property": "article:published_time"},
        {"name": "article:published_time"},
        {"name": "date"},
        {"name": "citation_publication_date"},
        {"name": "citation_date"},              # arXiv, e.g. "2026/10/28"
        {"itemprop": "datePublished"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and (parsed := _parse_iso_date(tag.get("content"))):
            return parsed
    time_tag = soup.find("time", attrs={"datetime": True})
    return _parse_iso_date(time_tag["datetime"]) if time_tag else None


def arxiv_search_query(query: str, categories: list[str]) -> str:
    """Abstract search for the first ARXIV_MAX_TERMS keywords (longer AND-queries time out)."""
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", query)
             if len(w) > 2 and w.lower() not in ARXIV_STOPWORDS][:ARXIV_MAX_TERMS]
    terms = " AND ".join(f"abs:{w}" for w in words)
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cats}) AND ({terms})" if terms else f"({cats})"


# --- Tools -------------------------------------------------------------------

def fetch_feeds(sources: list[dict], since: date) -> list[Article]:
    """Read the RSS/Atom feeds listed in sources.yaml and return entries published on or after `since`."""
    articles = []
    for source in sources:
        for feed_url in source.get("feeds", []):
            try:
                response = requests.get(feed_url, headers=HEADERS, timeout=20)
                response.raise_for_status()
            except requests.RequestException as exc:
                log.warning("feed failed: %s (%s)", feed_url, exc)
                continue
            parsed = feedparser.parse(response.content)
            articles.extend(entries_to_articles(parsed.entries, source, since))
    return articles


def is_listing_url(url: str) -> bool:
    """Search or listing pages (e.g. huggingface.co/papers?q=...) rather than articles."""
    return any(k.lower() in LISTING_PARAMS for k, _ in parse_qsl(urlparse(url).query))


def search_web(query: str, sources: list[dict], days: int, max_results: int = 8) -> list[Article]:
    """Recent web search via Tavily (news mode, last `days` days), restricted to the sources' domains.

    `published` is None when Tavily doesn't report a date; extract_full_text fills it
    from the page's meta tags.
    """
    from tavily import TavilyClient   # imported here so the rest of the module works without a key

    domains = [d for s in sources for d in s["domains"]]
    response = TavilyClient().search(
        query,
        topic="news",                 # general + start_date returned nothing in testing
        days=days,
        include_domains=domains,
        max_results=max_results,
    )
    articles = []
    for result in response.get("results", []):
        source = source_for_url(result["url"], sources)
        if source is None or is_listing_url(result["url"]):
            continue
        articles.append(Article(
            url=clean_url(result["url"]),
            title=" ".join(result.get("title", "").split()),
            source=source["name"],
            tier=source["tier"],
            published=_parse_iso_date(result.get("published_date")),
            snippet=" ".join(result.get("content", "").split())[:SNIPPET_CHARS],
        ))
    return articles


def search_arxiv(query: str, categories: list[str], since: date, max_results: int = 10, tier: int = 1) -> list[Article]:
    """Recent arXiv papers matching `query` in the given categories (newest first).

    Returns [] if arXiv is unavailable or still rate-limiting after one retry.
    """
    global _last_arxiv_call
    params = {
        "search_query": arxiv_search_query(query, categories),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    for attempt in range(2):
        wait = ARXIV_DELAY_SECONDS - (time.monotonic() - _last_arxiv_call)
        if wait > 0:
            time.sleep(wait)
        try:
            response = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=45)
            _last_arxiv_call = time.monotonic()
            if response.status_code == 429 and attempt == 0:
                time.sleep(ARXIV_RETRY_SECONDS)
                continue
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            _last_arxiv_call = time.monotonic()
            log.warning("arXiv search failed: %s (%s)", query, exc)
            return []
    source = {"name": "arXiv", "tier": tier}
    return entries_to_articles(feedparser.parse(response.content).entries, source, since)


def markdown_to_text(markdown: str) -> str:
    """Strip image and link markup from Tavily's markdown, keeping link text."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def extract_with_tavily(url: str) -> str | None:
    """Page text via Tavily extract, for sites that block direct requests (e.g. openai.com)."""
    if not os.getenv("TAVILY_API_KEY"):
        return None
    from tavily import TavilyClient

    try:
        results = TavilyClient().extract([url]).get("results", [])
    except Exception as exc:
        log.warning("tavily extract failed: %s (%s)", url, exc)
        return None
    return markdown_to_text(results[0].get("raw_content", "")) if results else None


def load_page(url: str, timeout: float = 20) -> dict | None:
    """Fetch a page and return {"text", "word_count", "published"}, or None if it can't be read.

    Falls back to Tavily extract when the site blocks the request or yields too little text.
    `published` is the date from the page's meta tags (None if absent or from the fallback).
    """
    text, published = None, None
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        if "html" in response.headers.get("Content-Type", ""):
            text = extract_main_text(response.text)
            published = page_published_date(response.text)
    except requests.RequestException as exc:
        log.info("direct fetch failed, trying fallback: %s (%s)", url, exc)
    if not text or len(text.split()) < MIN_PAGE_WORDS:
        text = extract_with_tavily(url)
    if not text or len(text.split()) < MIN_PAGE_WORDS:
        log.warning("no readable text: %s", url)
        return None
    return {"text": text, "word_count": len(text.split()), "published": published}


def url_resolves(url: str, timeout: float = 10) -> bool:
    """True unless the link is broken (404/410, 5xx, or no connection).

    401/403/429 count as resolving: the page exists but blocks scripts (e.g. openai.com).
    """
    try:
        response = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if response.status_code in (403, 405) or response.status_code >= 500:
            response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
            response.close()
        return response.status_code < 400 or response.status_code in BOT_BLOCK_STATUSES
    except requests.RequestException:
        return False


AGENT_TOOLS = [tool(f) for f in (fetch_feeds, search_web, search_arxiv, load_page)]
