"""Shared fixture: an isolated agent with fake LLM, embeddings, and network tools."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import config
import nodes
from fakes import FakeLLM, make_articles


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, path in {"DATA_DIR": tmp_path / "data", "HISTORY_DB": tmp_path / "data" / "history.db",
                       "CHECKPOINT_DB": tmp_path / "data" / "checkpoints.db",
                       "VECTOR_STORE_PATH": tmp_path / "data" / "vectors.json",
                       "BRIEFS_DIR": tmp_path / "briefs"}.items():
        monkeypatch.setattr(config, name, path)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    fake = FakeLLM()
    monkeypatch.setattr(nodes, "structured", fake)
    embeddings = DeterministicFakeEmbedding(size=32)
    monkeypatch.setattr(nodes, "embeddings_or_none", lambda: embeddings)

    feed = {"articles": make_articles()}
    monkeypatch.setattr(nodes.tools, "fetch_feeds", lambda sources, since: list(feed["articles"]))
    monkeypatch.setattr(nodes.tools, "search_arxiv", lambda *a, **k: [])
    broken, no_text = set(), set()
    monkeypatch.setattr(nodes.tools, "url_resolves", lambda url: url not in broken)
    monkeypatch.setattr(nodes.tools, "load_page", lambda url: (
        None if any(s in url for s in no_text) else {"text": "word " * 920, "word_count": 920, "published": None}))
    return {"llm": fake, "feed": feed, "broken": broken, "no_text": no_text, "tmp": tmp_path}
