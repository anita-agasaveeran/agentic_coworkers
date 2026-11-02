"""Shared fixture: an isolated agent with fake LLM, embeddings, and network tools."""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import config
import nodes
from fakes import FakeLLM, make_articles

# Fixed profile so tests don't depend on the real profile.json.
TEST_PROFILE = {
    "name": "Test",
    "career_goals": ["ML engineer internship"],
    "interests": {
        "llm agents and multi-agent systems": 1.0,
        "retrieval-augmented generation": 0.8,
        "recommender systems": 0.7,
        "cloud ML infrastructure": 0.5,
    },
    "skill_levels": {"python": "advanced"},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, path in {"DATA_DIR": tmp_path / "data", "HISTORY_DB": tmp_path / "data" / "history.db",
                       "CHECKPOINT_DB": tmp_path / "data" / "checkpoints.db",
                       "VECTOR_STORE_PATH": tmp_path / "data" / "vectors.json",
                       "BRIEFS_DIR": tmp_path / "briefs"}.items():
        monkeypatch.setattr(config, name, path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(TEST_PROFILE))
    monkeypatch.setattr(config, "PROFILE_PATH", profile_path)
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
