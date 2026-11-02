"""Fake LLM, postings, and scripted answers shared by the pipeline tests."""

import random
import re
from datetime import date

from state import (CriterionScore, Finding, GeneratedQuestion, Grade, OpportunityQueries, OpportunityQuery,
                   PostingAnalysis, PrepPlanOutput, ResourcePick, SkillRequirement, StrengthsGaps, TaskDraft)

SESSION_DATE = date(2026, 11, 1)

TEST_PROFILE = {
    "name": "Test",
    "targets": [
        {"role": "ML engineer intern", "type": "internship", "timeline": "summer 2027", "priority": 1.0},
        {"role": "ML for astronomy research assistant", "type": "research", "timeline": None, "priority": 0.8},
    ],
    "constraints": {},
    "skills": {"python": "advanced", "rag": "beginner"},
    "projects": [],
}


def req(skill, level, required, evidence):
    return SkillRequirement(skill=skill, level=level, required=required, evidence=evidence)


# url -> (search result type or None, PostingAnalysis)
POSTINGS = {
    "https://job-boards.greenhouse.io/acme/jobs/1": ("internship", PostingAnalysis(
        is_posting=True, title="ML Engineer Intern", type="internship", org="Acme", location="San Jose, CA",
        deadline="2026-12-15", closed=False, eligible=True, responsibilities=["Build RAG pipelines"],
        requirements=[req("rag", 2, True, "Experience building RAG systems"),
                      req("LLM agents", 2, True, "Built LLM agents with tool use"),
                      req("python", 2, True, "Strong Python"),
                      req("Fortran", 1, False, "Fortran is a plus")])),
    "https://jobs.lever.co/beta/0f6a2c1e-1b2c-4d5e-8f90-123456789abc": ("internship", PostingAnalysis(
        is_posting=True, title="AI Intern", type="internship", org="Beta", location=None, deadline=None,
        closed=False, eligible=True, responsibilities=["Design services"],
        requirements=[req("system_design", 2, True, "Design scalable services"),
                      req("rag", 1, False, "Familiar with retrieval")])),
    "https://job-boards.greenhouse.io/gamma/jobs/2": ("internship", PostingAnalysis(
        is_posting=True, title="Closed Intern", type="internship", org="Gamma", location=None, deadline=None,
        closed=True, eligible=True, responsibilities=[], requirements=[])),
    "https://www.nsf.gov/funding/reu": ("research", PostingAnalysis(
        is_posting=False, title=None, type="research", org="NSF", location=None, deadline=None,
        closed=False, eligible=True, responsibilities=[], requirements=[])),
    "https://example.org/my-posting": (None, PostingAnalysis(
        is_posting=True, title="Research Assistant, ML for Astronomy", type="research", org="Example Lab",
        location=None, deadline="2026-12-01", closed=False, eligible=True, responsibilities=["Fit models"],
        requirements=[req("ml for astronomy", 2, True, "Apply ML to astronomical surveys"),
                      req("python", 1, True, "Python")])),
}

POSTING_TEXT = {url: f"{a.title or 'Page'}\n{' '.join(r.evidence for r in a.requirements)}\n" + "details " * 200
                for url, (_, a) in POSTINGS.items()}


def search_results(query, opportunity_type, sources, days, max_results=8):
    return [{"url": url, "title": analysis.title or "Listing", "org": analysis.org, "type": t,
             "source": "Greenhouse", "snippet": "..."}
            for url, (t, analysis) in POSTINGS.items() if t == opportunity_type]


CODING = dict(
    starter_code='def add(a, b):\n    """Return a + b."""\n    raise NotImplementedError\n',
    tests=("from solution import add\n\n\ndef test_small():\n    assert add(1, 2) == 3\n\n\n"
           "def test_broken():\n    assert add(1, 1) == 3\n"),
    reference_solution="def add(a, b):\n    return a + b\n",
)
CORRECT_CODE = "def add(a, b):\n    return a + b\n"
WORDS = ["latency", "embedding", "cache", "shard", "gradient", "recall", "tokenizer", "agent", "index", "photon",
         "galaxy", "retry", "schema", "batch", "prior", "kernel", "queue", "vector", "rubric", "budget"]


class FakeLLM:
    """Stands in for structured(schema): answers each prompt type deterministically."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.prompts: list[str] = []
        self.coding_tests = CODING["tests"]

    def __call__(self, schema):
        llm = self

        class Runnable:
            def invoke(self, prompt):
                return llm.answer(schema, prompt)

            def batch(self, prompts, **kwargs):
                return [llm.answer(schema, p) for p in prompts]

        return Runnable()

    def count(self, name):
        return self.calls.get(name, 0)

    def answer(self, schema, prompt):
        self.calls[schema.__name__] = self.count(schema.__name__) + 1
        self.prompts.append(prompt)
        if schema is OpportunityQueries:
            return OpportunityQueries(queries=[OpportunityQuery(type="internship", query="ml intern 2027"),
                                               OpportunityQuery(type="research", query="astro ml reu"),
                                               OpportunityQuery(type="job", query="not a target type")])
        if schema is PostingAnalysis:
            url = re.search(r"^Page URL: (\S+)", prompt, flags=re.M).group(1)
            return POSTINGS[url][1]
        if schema is GeneratedQuestion:
            if "follow-up question" in prompt:
                words = " ".join(random.Random(-self.count("GeneratedQuestion")).choices(WORDS, k=12))
                return GeneratedQuestion(prompt=f"Follow-up #{self.count('GeneratedQuestion')}: why {words}?",
                                         grounding_id=None, reference_answer="Because.",
                                         rubric=["reasoning", "accuracy"], hints=["think"], starter_code=None,
                                         tests=None, reference_solution=None)
            kind = re.search(r"writing one (\w+) question", prompt).group(1)
            skill = re.search(r"^Skill: (\S+)", prompt, flags=re.M).group(1)
            extra = {**CODING, "tests": self.coding_tests} if kind == "coding" else \
                dict(starter_code=None, tests=None, reference_solution=None)
            return GeneratedQuestion(
                prompt=f"Q{self.count('GeneratedQuestion')} ({kind}) on {skill}: " + " ".join(
                    random.Random(self.count("GeneratedQuestion")).choices(WORDS, k=12)) + "?", grounding_id=1,
                reference_answer="A strong answer.", rubric=["correctness", "depth", "clarity"],
                hints=["hint one", "hint two", "hint three"], **extra)
        if schema is Grade:
            answer = prompt.split("Candidate's answer:\n", 1)[1]
            score = 5 if "strong" in answer else 2 if "weak" in answer else 4
            rubric = re.findall(r"^- (.+)$", prompt.split("Rubric criteria:\n", 1)[1].split("\n\n")[0], flags=re.M)
            return Grade(criteria=[CriterionScore(criterion=c, score=score) for c in rubric],
                         feedback="Feedback.", misconceptions=["mixes up X and Y"] if score < 3 else [])
        if schema is StrengthsGaps:
            qid = re.search(r"^- (\d{4}-\d\d-\d\d-q\d+)", prompt, flags=re.M)
            return StrengthsGaps(
                strengths=[Finding(skill="python", summary="Solid coding.",
                                   evidence=[qid.group(1) if qid else "R1", "q99"])],
                gaps=[Finding(skill="rag", summary="Retrieval design.", evidence=["R1"]),
                      Finding(skill="mlops", summary="Unsupported claim.", evidence=["made-up"])])
        if schema is PrepPlanOutput:
            return PrepPlanOutput(
                resources=[ResourcePick(id=1, skill="rag", why="Covers retrieval."),
                           ResourcePick(id=99, skill="rag", why="Not in the list.")],
                tasks=[TaskDraft(task="Read the RAG guide", skill="rag", resource_id=1, hours=3, due_in_days=3,
                                 done_when="Can explain chunking"),
                       TaskDraft(task="Build an agent", skill="llm_agents", resource_id=None, hours=30,
                                 due_in_days=40, done_when="Agent passes tests")])
        raise AssertionError(f"unexpected schema {schema}")


class ScriptedAnswers:
    """Answers interrupts: correct code for coding, then weak / strong text answers, "ok" for follow-ups."""

    def __init__(self, text_answers=("weak answer", "strong answer"), check_in=None):
        self.text_answers = list(text_answers)
        self.check_in = check_in or {}
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        if payload["kind"] == "check_in":
            return [{"id": t["id"], "status": self.check_in.get(t["task"], "open")} for t in payload["tasks"]]
        q = payload["question"]
        if q.get("parent_id"):
            return {"action": "answer", "answer": "ok answer", "hints_used": 0}
        if q["kind"] == "coding":
            return {"action": "answer", "answer": CORRECT_CODE, "hints_used": 1}
        answer = self.text_answers.pop(0) if self.text_answers else "ok answer"
        if answer in ("skip", "quit"):
            return {"action": answer, "answer": "", "hints_used": 0}
        return {"action": "answer", "answer": answer, "hints_used": 0}
