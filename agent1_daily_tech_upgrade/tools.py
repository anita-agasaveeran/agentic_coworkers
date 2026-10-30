"""External data tools: feeds, web search, arXiv, and page loading.

Nodes call these as plain functions. AGENT_TOOLS exposes the same functions
as LangChain tools in case a node is later turned into a tool-calling agent.
Every function returns metadata taken from the source response, never from an LLM.
"""

from datetime import date

from langchain_core.tools import tool

from state import Article


def fetch_feeds(sources: list[dict], since: date) -> list[Article]:
    """Read the RSS/Atom feeds listed in sources.yaml and return entries published on or after `since`."""
    # TODO: feedparser.parse(url) for each source's feeds; map entry.title, entry.link,
    #       entry.published_parsed, entry.summary -> Article; set source/tier from the source dict.
    raise NotImplementedError


def search_web(query: str, domains: list[str], days: int, max_results: int = 8) -> list[Article]:
    """Domain-restricted, recent web search via Tavily for sources without a usable feed."""
    # TODO: TavilyClient().search(query, topic="news", days=days, include_domains=domains,
    #       max_results=max_results); map results -> Article. Map the result URL's domain
    #       back to its source/tier in sources.yaml. Drop results without a published date.
    raise NotImplementedError


def search_arxiv(query: str, categories: list[str], since: date, max_results: int = 10) -> list[Article]:
    """Recent arXiv papers matching `query` in the given categories."""
    # TODO: GET http://export.arxiv.org/api/query?search_query=...&sortBy=submittedDate
    #       and parse the Atom response with feedparser. Use the abstract as the snippet.
    raise NotImplementedError


def load_page(url: str, timeout: float = 20) -> tuple[str, int] | None:
    """Fetch a page and return (main_text, word_count), or None if it can't be fetched or parsed."""
    # TODO: requests.get with a user agent; BeautifulSoup -> prefer <article>/<main>,
    #       strip nav/script/style; see load_web_page in LangGraph_LangChain_agents.ipynb.
    raise NotImplementedError


def url_resolves(url: str, timeout: float = 10) -> bool:
    """True if the URL answers with a 2xx/3xx status (link check before summarizing)."""
    # TODO: requests.head(..., allow_redirects=True); fall back to GET if HEAD is rejected.
    raise NotImplementedError


AGENT_TOOLS = [tool(f) for f in (fetch_feeds, search_web, search_arxiv, load_page)]
