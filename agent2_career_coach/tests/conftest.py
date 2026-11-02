"""Shared fixture: an isolated agent with a fake LLM, fake embeddings, and fake web tools."""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import agent
import config
import nodes
from fakes import POSTING_TEXT, TEST_PROFILE, FakeLLM, search_results


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, path in {"DATA_DIR": tmp_path / "data", "CAREER_DB": tmp_path / "data" / "career.db",
                       "CHECKPOINT_DB": tmp_path / "data" / "checkpoints.db",
                       "VECTOR_STORE_PATH": tmp_path / "data" / "vectors.json",
                       "REPORTS_DIR": tmp_path / "reports", "SESSIONS_DIR": tmp_path / "sessions",
                       "WORKSPACE_DIR": tmp_path / "workspace", "RESUME_PATH": tmp_path / "resume.md",
                       "AGENT1_HISTORY_DB": tmp_path / "agent1" / "history.db"}.items():
        monkeypatch.setattr(config, name, path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(TEST_PROFILE))
    monkeypatch.setattr(config, "PROFILE_PATH", profile_path)
    monkeypatch.setenv("TAVILY_API_KEY", "test")

    fake = FakeLLM()
    monkeypatch.setattr(nodes, "structured", fake)
    embeddings = DeterministicFakeEmbedding(size=32)
    monkeypatch.setattr(nodes, "embeddings_or_none", lambda: embeddings)

    unreadable = set()
    monkeypatch.setattr(nodes.tools, "search_opportunities", search_results)
    monkeypatch.setattr(nodes.tools, "load_posting", lambda url: None if url in unreadable else
                        {"text": POSTING_TEXT[url], "word_count": 210, "published": None, "can_apply": None})
    monkeypatch.setattr(nodes.tools, "search_resources", lambda query, sources, max_results=5: [
        {"url": f"https://docs.langchain.com/guide-{len(query)}", "title": f"Guide for {query}",
         "source": "LangChain / LangGraph docs", "snippet": "A guide."}])
    monkeypatch.setattr(nodes.tools, "url_resolves", lambda url: True)
    monkeypatch.setattr(nodes.tools, "search_arxiv", lambda *a, **k: [])

    def run(answers, session_date, **kwargs):
        monkeypatch.setattr(agent, "answer_interrupt", answers)
        return agent.run(session_date, **kwargs)

    return {"llm": fake, "tmp": tmp_path, "run": run, "unreadable": unreadable}
