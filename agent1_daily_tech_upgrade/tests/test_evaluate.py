"""Offline tests for evaluate.py, using runs saved by the fake-LLM pipeline."""

from datetime import date

import pytest

import agent
import evaluate
from state import FaithfulnessJudgment
from fakes import RUN_DATE, make_articles


def fake_judge(article, summary):
    if "recommender" in article["title"]:
        return FaithfulnessJudgment(score=3, unsupported_claims=["claims a 2x speedup"])
    return FaithfulnessJudgment(score=5, unsupported_claims=[])


@pytest.fixture
def two_runs(env, monkeypatch):
    def rate(payload):
        return [{"url": a["url"], "rating": r, "tags": []} for a, r in zip(payload["articles"], (5, 4, 2))]

    monkeypatch.setattr(agent, "ask_for_feedback", rate)
    agent.run(RUN_DATE, interactive=True)
    env["feed"]["articles"] = env["feed"]["articles"] + make_articles(day_offset=1)
    agent.run(date(2026, 10, 30), interactive=False)
    return env


def test_load_runs_reads_saved_checkpoints(two_runs):
    runs = evaluate.load_runs()
    assert [r["run_date"] for r in runs] == [RUN_DATE, date(2026, 10, 30)]
    assert all(len(r["selected"]) == 3 for r in runs)
    assert len(evaluate.load_runs(days=1)) == 1


def test_report_offline(two_runs):
    report = evaluate.evaluate(network=False, judge_fn=fake_judge)
    assert "Runs evaluated: 2 (2026-10-29 to 2026-10-30)" in report
    assert "| briefs with consistent references | 2/2 |" in report
    assert "| repeated articles | 0 |" in report
    assert "| precision at 3 | 2/3 (67%) |" in report
    assert "| mean rating 1 5 | 3.67 |" in report
    assert "| briefs within budget | 2/2 (100%) |" in report
    assert "| full text rate | 6/6 (100%) |" in report
    assert "| mean faithfulness 1 5 | 4.33 |" in report
    assert "| share scored 4 or 5 | 4/6 (67%) |" in report
    assert "LLM-based recommender systems: claims a 2x speedup" in report
    assert "| interests covered | 3/4 |" in report               # "rag evaluation" counts as RAG
    assert "| next focus under covered | cloud ML infrastructure |" in report


def test_report_skips_llm(two_runs):
    assert "| skipped | --no-llm |" in evaluate.evaluate(use_llm=False, network=False)


def test_brief_reference_consistency_detects_mismatch(two_runs):
    run = evaluate.load_runs()[0]
    assert evaluate.brief_references_consistent(run)
    run["brief_markdown"] = run["brief_markdown"].replace(run["selected"][0]["url"], "https://invented.example/x")
    assert not evaluate.brief_references_consistent(run)


def test_reference_accuracy_with_network_checks(two_runs, monkeypatch):
    runs = evaluate.load_runs()
    pages = {a["url"]: (f"{a['title']} | Blog", a["published"]) for r in runs for a in r["selected"]}
    first = runs[0]["selected"][0]["url"]
    pages[first] = (None, None)                      # blocked page: unverifiable
    monkeypatch.setattr(evaluate.tools, "url_resolves", lambda url: True)
    monkeypatch.setattr(evaluate, "page_metadata", lambda url: pages[url])
    result = evaluate.reference_accuracy(runs, network=True)
    assert result["links_resolving"] == "6/6 (100%)"
    assert result["titles_matching_page"] == "5/5 (100%)"
    assert result["pages_unverifiable"] == 1


@pytest.mark.parametrize("stored, page, expected", [
    ("The Art of Loop Engineering", "The Art of Loop Engineering | LangChain", True),
    ("Towards Reliable Generative Recommendation", "[2610.38378v1] Towards Reliable Generative Recommendation", True),
    ("The Art of Loop Engineering", "Pricing — LangChain", False),
])
def test_titles_match(stored, page, expected):
    assert evaluate.titles_match(stored, page) is expected
