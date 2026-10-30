"""End-to-end graph runs with a fake LLM, fake embeddings, and fake network tools."""

import json
import re
from datetime import date

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import agent
import config
import memory
import nodes
from state import CandidateScore, CandidateScores, QueryPlan, SummaryOutput

RUN_DATE = date(2026, 10, 29)
PUBLISHED = date(2026, 10, 28)

# title -> (relevance, depth, novelty, topics)
CATALOG = {
    "RAG evaluation at scale": ("Google Research", 1, (5, 5, 5, ["rag evaluation"])),
    "Multi-agent orchestration with LangGraph": ("AWS Machine Learning Blog", 1, (5, 4, 5, ["multi-agent systems"])),
    "Another RAG benchmark": ("Google Research", 1, (5, 5, 4, ["rag benchmarks"])),
    "LLM-based recommender systems": ("Hugging Face", 2, (5, 4, 4, ["recommender systems"])),
    "Funding announcement": ("OpenAI", 1, (1, 1, 2, ["company news"])),
    "Attention kernels on GPUs": ("NVIDIA Technical Blog", 2, (4, 5, 4, ["gpu kernels"])),
}


def slug(title: str) -> str:
    return re.sub(r"\W+", "-", title.lower()).strip("-")


def make_articles(day_offset: int = 0) -> list[dict]:
    return [{
        "url": f"https://example.org/{slug(title)}{'-v2' if day_offset else ''}",
        "title": title + (" (follow-up)" if day_offset else ""),
        "source": source, "tier": tier,
        "published": date.fromordinal(PUBLISHED.toordinal() + day_offset),
        "snippet": f"About {title}.",
    } for title, (source, tier, _) in CATALOG.items()]


class FakeLLM:
    """Stands in for structured(schema): answers each prompt type deterministically."""

    def __init__(self, scores_override=None):
        self.calls = {"QueryPlan": 0, "CandidateScores": 0, "SummaryOutput": 0}
        self.scores_override = scores_override

    def __call__(self, schema):
        llm = self

        class Runnable:
            def invoke(self, prompt):
                return llm.answer(schema, prompt)

            def batch(self, prompts):
                return [llm.answer(schema, p) for p in prompts]

        return Runnable()

    def answer(self, schema, prompt):
        self.calls[schema.__name__] += 1
        if schema is QueryPlan:
            return QueryPlan(queries=["rag evaluation", "multi-agent systems", "recommender systems"],
                             arxiv_queries=["agentic retrieval"])
        if schema is CandidateScores:
            scores = []
            for cid, title in re.findall(r"^\[(\d+)\] (.+?) \|", prompt, flags=re.M):
                base = title.removesuffix(" (follow-up)")
                rel, depth, nov, topics = self.scores_override or CATALOG[base][2]
                scores.append(CandidateScore(id=int(cid), relevance=rel, depth=depth, novelty=nov, topics=topics))
            return CandidateScores(scores=scores)
        title = re.search(r"^Article: (.+?) \(", prompt, flags=re.M).group(1)
        return SummaryOutput(
            summary=f"Summary of {title}.", key_takeaways=["one", "two", "three"],
            why_it_matters="Fits your goals.", think_about="What next?", reading_scope="whole article")


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


def test_full_run_writes_brief_and_memory(env):
    result = agent.run(RUN_DATE, interactive=False)

    titles = [a["title"] for a in result["selected"]]
    assert titles == ["RAG evaluation at scale", "Multi-agent orchestration with LangGraph",
                      "LLM-based recommender systems"]          # diversity skips the 2nd Google article
    assert env["llm"].calls["QueryPlan"] == 1                  # no retry needed

    brief = (env["tmp"] / "briefs" / "2026-10-29.md").read_text()
    assert brief == result["brief_markdown"]
    assert brief.startswith("# Daily Tech Brief — Oct 29, 2026   (≈")
    for a in result["selected"]:
        assert a["url"] in brief
    assert "Summary based on the feed snippet" not in brief
    minutes = [s["read_minutes"] for s in result["summaries"]]
    assert minutes == [4, 4, 4]                                 # 920 words / 230 wpm
    assert config.SESSION_MINUTES[0] <= config.BRIEF_MINUTES + sum(minutes) <= config.SESSION_MINUTES[1]

    conn = memory.connect()
    assert memory.seen_urls(conn) == {a["url"] for a in result["selected"]}
    assert config.VECTOR_STORE_PATH.exists()


def test_next_day_skips_articles_already_shown(env):
    agent.run(RUN_DATE, interactive=False)
    env["feed"]["articles"] = make_articles() + make_articles(day_offset=1)
    result = agent.run(date(2026, 10, 30), interactive=False)
    assert all(a["title"].endswith("(follow-up)") for a in result["selected"])


def test_snippet_only_article_replaced_when_full_text_backup_exists(env):
    env["no_text"].add("recommender")
    result = agent.run(RUN_DATE, interactive=False)
    titles = [a["title"] for a in result["selected"]]
    assert "LLM-based recommender systems" not in titles and "Attention kernels on GPUs" in titles
    assert all(a["text_source"] == "full_text" for a in result["selected"])


def test_snippet_only_article_used_when_no_alternative(env):
    env["no_text"].update({"recommender", "attention-kernels"})
    result = agent.run(RUN_DATE, interactive=False)
    snippet = [a for a in result["selected"] if a["text_source"] == "snippet"]
    assert [a["title"] for a in snippet] == ["LLM-based recommender systems"]
    assert "Summary based on the feed snippet" in result["brief_markdown"]
    assert result["summaries"][-1]["read_minutes"] == nodes.slot_minutes(3)


def test_low_scores_retry_then_publish_empty_brief(env):
    env["llm"].scores_override = (1, 1, 1, ["noise"])
    result = agent.run(RUN_DATE, interactive=False)
    assert env["llm"].calls["QueryPlan"] == 1 + config.MAX_RETRIES
    assert result["lookback_days"] == config.BROADENED_LOOKBACK_DAYS
    assert result["selected"] == []
    assert "No articles met today's quality bar" in result["brief_markdown"]


def test_broken_link_replaced_by_next_diverse_candidate(env):
    env["broken"].add("https://example.org/multi-agent-orchestration-with-langgraph")
    result = agent.run(RUN_DATE, interactive=False)
    titles = [a["title"] for a in result["selected"]]
    assert "Multi-agent orchestration with LangGraph" not in titles
    assert "Attention kernels on GPUs" in titles and len(titles) == 3


def test_feedback_is_stored_and_shapes_progress(env, monkeypatch):
    def rate(payload):
        first, second = payload["articles"][:2]
        return [{"url": first["url"], "rating": 5, "tags": ["more"]},
                {"url": second["url"], "rating": 1, "tags": ["less"]}]

    monkeypatch.setattr(agent, "ask_for_feedback", rate)
    agent.run(RUN_DATE, interactive=True)

    conn = memory.connect()
    assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 2
    progress = memory.learning_progress(conn, date(2026, 10, 30), config.load_profile_file()["interests"])
    assert "rag evaluation" in progress["liked_topics"]
    assert "multi-agent systems" in progress["disliked_topics"]
    assert progress["recent_topics"]["rag evaluation"] == 1


# --- Helpers ----------------------------------------------------------------

@pytest.mark.parametrize("url, expected", [
    ("http://www.Example.org/post/", "https://example.org/post"),
    ("https://arxiv.org/abs/2610.37962v1", "https://arxiv.org/abs/2610.37962"),
    ("https://arxiv.org/pdf/2610.37962v2", "https://arxiv.org/abs/2610.37962"),
])
def test_normalize_url(url, expected):
    assert nodes.normalize_url(url) == expected


@pytest.mark.parametrize("rec_score, expected", [
    (0.85, ["a", "c"]),     # within the margin of b (0.94): covers a second interest
    (0.70, ["a", "b"]),     # too far below b: keep the stronger agent article
])
def test_pick_diverse_spreads_across_interests_within_margin(rec_score, expected):
    interests = {"llm agents": 1.0, "recommender systems": 0.7}
    ranked = [
        {"url": "a", "source": "S1", "topics": ["agent loops"], "total_score": 0.95},
        {"url": "b", "source": "S2", "topics": ["agent tools"], "total_score": 0.94},
        {"url": "c", "source": "S3", "topics": ["recommender systems"], "total_score": rec_score},
    ]
    assert [a["url"] for a in nodes.pick_diverse(ranked, 2, 0.6, interests)] == expected
    assert [a["url"] for a in nodes.pick_diverse(ranked, 2, 0.6)] == ["a", "b"]


def test_recency_score():
    assert nodes.recency_score(RUN_DATE, RUN_DATE, 3) == 1.0
    assert nodes.recency_score(date(2026, 10, 26), RUN_DATE, 3) == 0.0
    assert nodes.recency_score(None, RUN_DATE, 3) == 0.5


def test_under_covered_interests(env):
    conn = memory.connect()
    memory.record_brief(conn, date(2026, 10, 28), [{
        "url": "u", "title": "t", "source": "s", "published": PUBLISHED,
        "topics": ["retrieval-augmented generation"], "total_score": 0.9}])
    interests = {"retrieval-augmented generation": 0.8, "recommender systems": 0.7, "llm agents": 1.0}
    progress = memory.learning_progress(conn, RUN_DATE, interests)
    assert progress["under_covered"] == ["llm agents", "recommender systems"]
    assert json.loads(conn.execute("SELECT topics FROM articles_seen").fetchone()[0]) == \
        ["retrieval-augmented generation"]
