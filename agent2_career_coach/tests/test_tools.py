"""Offline tests for tools.py (no network; run_tests starts a local pytest subprocess)."""

import sys
import types
from datetime import date

import pytest

import config
import tools

SOURCES = config.load_opportunity_sources()["opportunity_sources"]
RESOURCE_SOURCES = config.load_opportunity_sources()["resource_sources"]
TODAY = date(2026, 11, 1)


def fake_tavily(monkeypatch, results):
    calls = []

    class FakeClient:
        def search(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            return {"results": results}

    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeClient))
    return calls


def source(name):
    return next(s for s in SOURCES if s["name"] == name)


@pytest.mark.parametrize("url, expected", [
    ("http://job-boards.greenhouse.io/scaleai/jobs/4735665005", "https://job-boards.greenhouse.io/scaleai/jobs/4735665005"),
    ("https://jobs.lever.co/acme/x?utm_source=feed&lever-origin=applied#apply", "https://jobs.lever.co/acme/x?lever-origin=applied"),
])
def test_canonical_url(url, expected):
    assert tools.canonical_url(url) == expected


@pytest.mark.parametrize("title, expected", [
    ("Job Application for Machine Learning Intern (Summer 2027) at Klaviyo Campus",
     ("Machine Learning Intern (Summer 2027)", "Klaviyo Campus")),
    ("Machine Learning Engineer Intern - Myworkdayjobs.com", ("Machine Learning Engineer Intern", None)),
    ("AI Engineer Intern -  Fury Team (Summer 2027)", ("AI Engineer Intern - Fury Team (Summer 2027)", None)),
])
def test_posting_title(title, expected):
    assert tools.posting_title(title) == expected


@pytest.mark.parametrize("name, url, expected", [
    ("Greenhouse", "https://job-boards.greenhouse.io/applovin/jobs/4713038006", True),
    ("Greenhouse", "https://job-boards.greenhouse.io/applovin", False),
    ("Lever", "https://jobs.lever.co/acme/0f6a2c1e-1b2c-4d5e-8f90-123456789abc", True),
    ("Lever", "https://jobs.lever.co/acme?department=Engineering", False),
    ("Ashby", "https://jobs.ashbyhq.com/geniusai/06198681-5c9d-40fa-aa68-ac92b65227c4", True),
    ("Workday", "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Intern_JR2027170", True),
    ("Workday", "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite", False),
    ("NSF REU", "https://www.nsf.gov/funding/initiatives/reu/students", True),
    ("NSF REU", "https://www.nsf.gov/search?q=reu", False),
    ("Y Combinator", "https://www.ycombinator.com/companies/acme/jobs/123-ml-engineer", True),
    ("Y Combinator", "https://www.ycombinator.com/companies/industry/machine-learning", False),
])
def test_is_posting_url(name, url, expected):
    assert tools.is_posting_url(url, source(name)) is expected


def test_search_opportunities_searches_only_sources_for_the_type(monkeypatch):
    calls = fake_tavily(monkeypatch, [
        {"url": "http://job-boards.greenhouse.io/klaviyocampus/jobs/7999274003",
         "title": "Job Application for ML Engineer Internship-Summer 2027 at Klaviyo Campus",
         "content": "As a  Machine Learning\nEngineer Intern"},
        {"url": "https://job-boards.greenhouse.io/klaviyocampus/jobs/7999274003", "title": "duplicate"},
        {"url": "https://job-boards.greenhouse.io/klaviyocampus", "title": "Klaviyo board"},
        {"url": "https://jobs.ashbyhq.com/scout-ai/3b639e39-abd7-4a0c-9bea-e1cd0a72db99",
         "title": "AI Engineer Intern (Summer 2027)", "content": "We're seeking"},
        {"url": "https://www.linkedin.com/jobs/view/1", "title": "Off the allow-list"},
    ])
    found = tools.search_opportunities("ml intern summer 2027", "internship", SOURCES, days=30)

    domains = calls[0]["include_domains"]
    assert "jobs.lever.co" in domains and "intern.nasa.gov" in domains
    assert "nsf.gov" not in domains and "ycombinator.com" not in domains
    assert calls[0]["topic"] == "general" and calls[0]["time_range"] == "month"
    assert found == [
        {"url": "https://job-boards.greenhouse.io/klaviyocampus/jobs/7999274003",
         "title": "ML Engineer Internship-Summer 2027", "org": "Klaviyo Campus", "type": "internship",
         "source": "Greenhouse", "snippet": "As a Machine Learning Engineer Intern"},
        {"url": "https://jobs.ashbyhq.com/scout-ai/3b639e39-abd7-4a0c-9bea-e1cd0a72db99",
         "title": "AI Engineer Intern (Summer 2027)", "org": None, "type": "internship",
         "source": "Ashby", "snippet": "We're seeking"},
    ]


def test_search_opportunities_without_sources_for_the_type(monkeypatch):
    calls = fake_tavily(monkeypatch, [])
    only_jobs = [source("Lever")]
    assert tools.search_opportunities("reu astronomy", "research", only_jobs, days=7) == []
    assert calls == []


def test_search_resources(monkeypatch):
    calls = fake_tavily(monkeypatch, [
        {"url": "https://docs.langchain.com/oss/python/langgraph/interrupts#resume",
         "title": "Interrupts - Docs by LangChain", "content": "Pause a graph"},
        {"url": "https://huggingface.co/search?q=interrupts", "title": "Search"},
        {"url": "https://medium.com/x", "title": "Off the allow-list"},
    ])
    found = tools.search_resources("langgraph interrupts", RESOURCE_SOURCES)
    assert calls[0]["topic"] == "general" and "docs.langchain.com" in calls[0]["include_domains"]
    assert found == [{"url": "https://docs.langchain.com/oss/python/langgraph/interrupts",
                      "title": "Interrupts - Docs by LangChain", "source": "LangChain / LangGraph docs",
                      "snippet": "Pause a graph"}]


@pytest.mark.parametrize("text, deadline, expected", [
    ("Apply now. Applications due December 1.", date(2026, 12, 1), True),
    ("Apply now.", None, True),
    ("Apply now.", date(2026, 10, 31), False),
    ("We are no longer accepting applications for this role.", None, False),
    ("This position has been filled.", None, False),
    ("Applications are now closed for Summer 2027.", None, False),
    ("Sorry, we couldn't find anything here", None, False),
    ("The job is no longer available.", None, False),
    ("Remote applications are welcome; the office is closed on Fridays.", None, True),
])
def test_posting_is_open(text, deadline, expected):
    assert tools.posting_is_open({"text": text, "can_apply": None}, deadline, TODAY) is expected


def test_posting_is_open_uses_the_job_board_flag():
    assert not tools.posting_is_open({"text": "Apply now.", "can_apply": False}, None, TODAY)
    assert tools.posting_is_open({"text": "Apply now.", "can_apply": True}, None, TODAY)


WORKDAY_URL = ("https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/"
               "US-TX-Austin/AI-Intern---Summer-2027_JR2027170")


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise tools.requests.HTTPError(f"{self.status}")

    def json(self):
        return self.data


def test_load_posting_reads_workday_json(monkeypatch):
    urls = []
    info = {"title": "AI Intern - Summer 2027", "location": "US, TX, Austin", "startDate": "2026-10-30",
            "canApply": True, "jobDescription": "<p>Build agents.</p><ul><li>Python</li><li>PyTorch</li></ul>"}
    monkeypatch.setattr(tools.requests, "get",
                        lambda url, **k: urls.append(url) or FakeResponse({"jobPostingInfo": info}))
    posting = tools.load_posting(WORKDAY_URL)
    assert urls == ["https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/job/"
                    "US-TX-Austin/AI-Intern---Summer-2027_JR2027170"]
    assert posting["text"] == "AI Intern - Summer 2027\nUS, TX, Austin\nBuild agents.\nPython\nPyTorch"
    assert posting["published"] == date(2026, 10, 30) and posting["can_apply"] is True


def test_load_posting_falls_back_to_the_page(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda url, **k: FakeResponse({"message": "denied"}, 403))
    page = {"text": "posting text", "word_count": 2, "published": None}
    monkeypatch.setattr(tools, "load_page", lambda url, timeout=20: page)
    assert tools.load_posting(WORKDAY_URL) == {**page, "can_apply": None}
    assert tools.load_posting("https://jobs.lever.co/acme/x") == {**page, "can_apply": None}
    monkeypatch.setattr(tools, "load_page", lambda url, timeout=20: None)
    assert tools.load_posting("https://jobs.lever.co/acme/x") is None


TESTS = """
from solution import add

def test_small():
    assert add(1, 2) == 3

def test_negative():
    assert add(-1, -2) == -3
"""


def test_run_tests_all_pass():
    result = tools.run_tests("def add(a, b):\n    return a + b\n", TESTS)
    assert result["results"] == {"test_small": True, "test_negative": True}
    assert (result["passed"], result["total"], result["pass_rate"]) == (2, 2, 1.0)
    assert not result["timed_out"]


def test_run_tests_partial_pass():
    result = tools.run_tests("def add(a, b):\n    return abs(a) + abs(b)\n", TESTS)
    assert result["results"] == {"test_small": True, "test_negative": False}
    assert result["pass_rate"] == 0.5
    assert "test_negative" in result["output"]


def test_run_tests_broken_solution_scores_zero():
    result = tools.run_tests("def add(a, b)\n    return a + b\n", TESTS)
    assert result["passed"] == 0 and result["pass_rate"] == 0.0
    assert "SyntaxError" in result["output"]


def test_run_tests_times_out():
    result = tools.run_tests("def add(a, b):\n    while True:\n        pass\n", TESTS, timeout=3)
    assert result["timed_out"] and result["pass_rate"] == 0.0


def test_run_tests_hides_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    tests = "import os\nfrom solution import f\n\ndef test_env():\n    assert 'ANTHROPIC_API_KEY' not in os.environ\n"
    assert tools.run_tests("def f():\n    pass\n", tests)["pass_rate"] == 1.0
