"""Offline tests for tools.py (no network)."""

import sys
import types
from datetime import date

import feedparser
import pytest

import tools

SOURCES = [
    {"name": "Google Research", "tier": 1, "domains": ["research.google", "blog.google"]},
    {"name": "AWS Machine Learning Blog", "tier": 1, "domains": ["aws.amazon.com"]},
]

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Blog</title>
  <item><title>  New   RAG method </title><link>https://research.google/blog/new-rag</link>
    <pubDate>Thu, 29 Oct 2026 17:00:00 GMT</pubDate>
    <description>&lt;p&gt;We present &lt;b&gt;a method&lt;/b&gt;.&lt;/p&gt;</description></item>
  <item><title>Old post</title><link>https://research.google/blog/old</link>
    <pubDate>Mon, 01 Sep 2026 10:00:00 GMT</pubDate></item>
  <item><title>Undated post</title><link>https://research.google/blog/undated</link></item>
</channel></rss>"""

ARTICLE_HTML = """<html><head>
  <meta property="article:published_time" content="2026-10-28T09:30:00Z">
</head><body>
  <nav>Home | Blog</nav>
  <article><h1>Title</h1><p>{body}</p><script>track()</script></article>
  <footer>Copyright</footer>
</body></html>"""


def test_entries_to_articles_keeps_recent_dated_entries():
    entries = feedparser.parse(RSS).entries
    articles = tools.entries_to_articles(entries, SOURCES[0], since=date(2026, 10, 26))
    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "New RAG method"
    assert a["url"] == "https://research.google/blog/new-rag"
    assert a["published"] == date(2026, 10, 29)
    assert a["source"] == "Google Research" and a["tier"] == 1
    assert a["snippet"] == "We present a method ."


@pytest.mark.parametrize("url, expected", [
    ("https://research.google/blog/x", "Google Research"),
    ("https://www.blog.google/technology/ai/x", "Google Research"),
    ("https://aws.amazon.com/blogs/machine-learning/x", "AWS Machine Learning Blog"),
    ("https://notamazon.com/x", None),
    ("https://evil-research.google.com/x", None),
])
def test_source_for_url(url, expected):
    source = tools.source_for_url(url, SOURCES)
    assert (source["name"] if source else None) == expected


def test_extract_main_text_drops_page_chrome():
    text = tools.extract_main_text(ARTICLE_HTML.format(body="Body text."))
    assert "Body text." in text and "Title" in text
    assert "Home" not in text and "Copyright" not in text and "track()" not in text


def test_extract_main_text_ignores_related_post_cards():
    html = ("<html><body><main><article>Related: other post</article>"
            "<div>" + "real body words " * 50 + "</div></main></body></html>")
    text = tools.extract_main_text(html)
    assert "real body words" in text


def test_clean_url_strips_tracking():
    assert tools.clean_url("https://research.ibm.com/blog/x?utm_medium=rss&utm_source=rss&id=3#top") \
        == "https://research.ibm.com/blog/x?id=3"


def _response(status, **kwargs):
    import requests
    r = requests.Response()
    r.status_code = status
    r._content = kwargs.get("content", b"")
    r.close = lambda: None
    return r


@pytest.mark.parametrize("status, expected", [(200, True), (403, True), (404, False), (503, False)])
def test_url_resolves(monkeypatch, status, expected):
    monkeypatch.setattr(tools.requests, "head", lambda *a, **k: _response(status))
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _response(status))
    assert tools.url_resolves("https://example.com/x") is expected


def test_search_arxiv_retries_once_after_429(monkeypatch):
    atom = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>A paper</title><link href="http://arxiv.org/abs/2610.00001v1"/>
        <published>2026-10-28T10:00:00Z</published><summary>Abstract.</summary></entry></feed>"""
    responses = iter([_response(429), _response(200, content=atom)])
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: next(responses))
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    papers = tools.search_arxiv("agents", ["cs.AI"], since=date(2026, 10, 26))
    assert [(p["title"], p["source"], p["published"]) for p in papers] == [("A paper", "arXiv", date(2026, 10, 28))]


def test_page_published_date_from_meta():
    assert tools.page_published_date(ARTICLE_HTML.format(body="x")) == date(2026, 10, 28)
    assert tools.page_published_date("<html><body>no date</body></html>") is None


@pytest.mark.parametrize("value, expected", [
    ("2026-10-29", date(2026, 10, 29)),
    ("2026-10-29T15:00:00Z", date(2026, 10, 29)),
    ("Thu, 29 Oct 2026 17:00:00 GMT", date(2026, 10, 29)),
    ("not a date", None),
    (None, None),
])
def test_parse_iso_date(value, expected):
    assert tools._parse_iso_date(value) == expected


def test_arxiv_search_query():
    q = tools.arxiv_search_query("RAG evaluation for LLM agents", ["cs.AI", "cs.CL"])
    assert q == "(cat:cs.AI OR cat:cs.CL) AND (all:RAG AND all:evaluation AND all:LLM AND all:agents)"


def test_load_page(monkeypatch):
    html = ARTICLE_HTML.format(body="word " * 200)
    response = types.SimpleNamespace(
        text=html, headers={"Content-Type": "text/html; charset=utf-8"}, raise_for_status=lambda: None)
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: response)
    page = tools.load_page("https://research.google/blog/x")
    assert page["word_count"] > 200 and page["published"] == date(2026, 10, 28)

    response.text = ARTICLE_HTML.format(body="too short")
    assert tools.load_page("https://research.google/blog/x") is None


def test_search_web_maps_results_and_restricts_domains(monkeypatch):
    calls = {}

    class FakeClient:
        def search(self, query, **kwargs):
            calls.update(kwargs)
            return {"results": [
                {"url": "https://aws.amazon.com/blogs/machine-learning/a", "title": "A",
                 "content": "About A", "published_date": "Wed, 28 Oct 2026 12:00:00 GMT"},
                {"url": "https://research.google/blog/b", "title": "B", "content": "About B"},
                {"url": "https://elsewhere.com/c", "title": "C", "content": "About C"},
            ]}

    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeClient))
    articles = tools.search_web("rag", SOURCES, since=date(2026, 10, 26))

    assert calls["include_domains"] == ["research.google", "blog.google", "aws.amazon.com"]
    assert calls["start_date"] == "2026-10-26"
    assert [a["source"] for a in articles] == ["AWS Machine Learning Blog", "Google Research"]
    assert articles[0]["published"] == date(2026, 10, 28)
    assert articles[1]["published"] is None
