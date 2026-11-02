"""Fake LLM and article catalog shared by the pipeline and evaluation tests."""

import re
from datetime import date

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
        self.interests: dict[str, str] = {}   # title -> interest label the fake scorer returns
        self.prompts: list[str] = []

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
        self.prompts.append(prompt)
        if schema is QueryPlan:
            return QueryPlan(queries=["rag evaluation", "multi-agent systems", "recommender systems"],
                             arxiv_queries=["agentic retrieval"])
        if schema is CandidateScores:
            scores = []
            for cid, title in re.findall(r"^\[(\d+)\] (.+?) \|", prompt, flags=re.M):
                base = title.removesuffix(" (follow-up)")
                rel, depth, nov, topics = self.scores_override or CATALOG[base][2]
                scores.append(CandidateScore(id=int(cid), relevance=rel, depth=depth, novelty=nov, topics=topics,
                                             interest=self.interests.get(base, "none")))
            return CandidateScores(scores=scores)
        title = re.search(r"^Article: (.+?) \(", prompt, flags=re.M).group(1)
        return SummaryOutput(
            summary=f"Summary of {title}.", key_takeaways=["one", "two", "three"],
            why_it_matters="Fits your goals.", think_about="What next?", reading_scope="whole article")
