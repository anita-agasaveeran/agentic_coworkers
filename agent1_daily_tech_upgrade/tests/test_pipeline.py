"""End-to-end graph runs with a fake LLM, fake embeddings, and fake network tools."""

import json
from datetime import date

import pytest

import agent
import config
import memory
import nodes
from fakes import PUBLISHED, RUN_DATE, make_articles


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


def test_scorer_interest_label_drives_coverage(env):
    # Keyword matching can't tie "gpu kernels" to "cloud ML infrastructure"; the scorer's label can.
    env["llm"].interests["Attention kernels on GPUs"] = "cloud ML infrastructure"
    env["llm"].interests["RAG evaluation at scale"] = "Retrieval-Augmented Generation"   # case-insensitive
    agent.run(RUN_DATE, interactive=False)
    conn = memory.connect()
    tags = {r["title"]: json.loads(r["topics"]) for r in conn.execute("SELECT title, topics FROM articles_seen")}
    assert tags["RAG evaluation at scale"] == ["rag evaluation", "retrieval-augmented generation"]
    progress = memory.learning_progress(conn, date(2026, 10, 30), config.load_profile_file()["interests"])
    assert "retrieval-augmented generation" not in progress["under_covered"]


def test_rate_saved_brief_later(env, monkeypatch):
    agent.run(RUN_DATE, interactive=False)
    answers = iter(["x", "5", "more", "", "2", ""])     # invalid input re-prompts; Enter skips
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    feedback = agent.rate(RUN_DATE)
    assert [f["rating"] for f in feedback] == [5, 2] and feedback[0]["tags"] == ["more"]

    conn = memory.connect()
    assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 2
    answers = iter(["4", ""])                           # only the skipped article is asked again
    assert [f["rating"] for f in agent.rate(RUN_DATE)] == [4]
    assert agent.rate(date(2026, 1, 1)) == []           # no brief saved for that day


def test_ratings_reach_the_scoring_prompt(env, monkeypatch):
    def rate(payload):
        first, second = payload["articles"][:2]
        return [{"url": first["url"], "rating": 5, "tags": ["more"]},
                {"url": second["url"], "rating": 1, "tags": ["less"]}]

    monkeypatch.setattr(agent, "ask_for_feedback", rate)
    agent.run(RUN_DATE, interactive=True)
    env["feed"]["articles"] = make_articles(day_offset=1)
    env["llm"].prompts.clear()
    agent.run(date(2026, 10, 30), interactive=False)
    scoring = next(p for p in env["llm"].prompts if p.startswith("Score each candidate"))
    assert "topics they liked: rag evaluation;" in scoring
    assert "topics they want less of: multi-agent systems." in scoring


def test_tags_skip_profile_interests(env):
    # A "less" on one cosmology paper shouldn't demote the whole interest it was picked for.
    conn = memory.connect()
    articles = [
        {"url": "a", "title": "tSZ inference", "source": "arXiv", "published": PUBLISHED,
         "topics": ["thermal sz effect"], "interest": "ML for Physics", "total_score": 0.9},
        {"url": "b", "title": "Dual AGN", "source": "Nature", "published": PUBLISHED,
         "topics": ["dual agn"], "interest": "astrophysics", "total_score": 0.9},
    ]
    memory.record_brief(conn, date(2026, 10, 28), articles)
    memory.record_feedback(conn, date(2026, 10, 28), [
        {"url": "a", "rating": 2, "tags": ["less"]},
        {"url": "b", "rating": 5, "tags": ["more"]},
    ])
    progress = memory.learning_progress(conn, RUN_DATE, {"ml for physics": 0.9, "astrophysics": 1.0})
    assert progress["liked_topics"] == ["dual agn"]
    assert progress["disliked_topics"] == ["thermal sz effect"]
