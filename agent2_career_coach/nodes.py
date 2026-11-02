"""Graph nodes for nextLevelPersonalCoachForCareerLevel (one function per node in ARCHITECTURE.md §4).

Each node takes the current CoachState and returns a partial state update.
"""

import logging
from typing import Literal

from langgraph.types import interrupt

import config
from state import CoachState, Question

log = logging.getLogger(__name__)


# --- Helpers -----------------------------------------------------------------

def structured(schema):
    """The chat model constrained to `schema` via native structured outputs (no forced tool use)."""
    return config.get_llm().with_structured_output(schema, method="json_schema")


def embeddings_or_none():
    """Embeddings model, or None when it isn't configured (repeat checks then use exact text only)."""
    try:
        return config.get_embeddings()
    except Exception as exc:   # missing key or provider package
        log.warning("embeddings unavailable, checking repeats by exact text only (%s)", exc)
        return None


def top_level_questions(questions: list[Question]) -> list[Question]:
    return [q for q in questions if not q.get("parent_id")]


def follow_ups_of(questions: list[Question], question: Question) -> int:
    """Follow-ups already asked on the top-level question that `question` belongs to."""
    root = question.get("parent_id") or question["id"]
    return sum(1 for q in questions if q.get("parent_id") == root)


def follow_up_mode(score: float) -> Literal["probe", "challenge"] | None:
    if score < config.PROBE_BELOW:
        return "probe"
    if score >= config.CHALLENGE_AT:
        return "challenge"
    return None


def public_question(question: Question) -> dict:
    """What I see: no reference answer or tests. Hints are revealed by the CLI on request."""
    return {k: question.get(k) for k in
            ("id", "skill", "kind", "difficulty", "prompt", "hints", "workspace_file", "parent_id")}


# --- Scoring (ARCHITECTURE.md §7) --------------------------------------------

def answer_score(kind: str, criteria: dict[str, int], tests_passed: float | None, hints_used: int) -> float:
    """0-1 answer score: mean criteria (coding: CODING_WEIGHTS with test pass rate), minus HINT_PENALTY per hint."""
    raise NotImplementedError


def readiness_from_history(history: list[dict], self_rated: dict[str, str], reading: dict[str, dict],
                           today) -> tuple[dict[str, float], set[str]]:
    """Per-skill readiness and the set of skills with at least one assessment.

    Recency-weighted mean (READINESS_HALF_LIFE_DAYS) of score x DIFFICULTY_FACTOR; self-rated
    fallback via SELF_RATED_READINESS (matched on the level's first word); plus up to READING_BONUS.
    """
    raise NotImplementedError


def skill_demand(opportunities: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """(demand, target level) per skill across the top MAX_OPPORTUNITIES by fit, demand normalized to max 1."""
    raise NotImplementedError


def skill_gaps(demand: dict[str, float], targets: dict[str, int], readiness: dict[str, float]) -> dict[str, float]:
    """gap = demand x max(0, TARGET_READINESS[target] - readiness), largest first."""
    raise NotImplementedError


def role_readiness(target: dict, opportunities: list[dict], readiness: dict[str, float]) -> dict:
    """{"readiness": 0-1, "boost": [three skills that would raise it most]} for one profile target."""
    raise NotImplementedError


def difficulty_for(readiness: float) -> int:
    """Question difficulty 1-3 from DIFFICULTY_BANDS."""
    raise NotImplementedError


# --- Nodes -------------------------------------------------------------------

def load_profile(state: CoachState) -> dict:
    """Load profile.json, resume.md, stored opportunities, open plan tasks, and Agent 1 reading.

    Returns: {"profile": {..., "resume": str | None, "reading": memory.agent1_reading(...)},
              "opportunities": memory.stored_opportunities(...), "open_tasks": [...],
              "discovery_due": refresh or no opportunities or newest older than OPPORTUNITY_MAX_AGE_DAYS,
              "questions": [], "assessments": [], "quit": False}
    """
    raise NotImplementedError


def check_in_plan(state: CoachState) -> dict:
    """Pause to mark each open plan task done / partial / open / dropped (human-in-the-loop)."""
    tasks = state.get("open_tasks", [])
    if not state.get("interactive", True) or not tasks:
        return {"plan_updates": []}
    updates = interrupt({
        "kind": "check_in",
        "tasks": [{"id": t["id"], "task": t["task"], "due": str(t.get("due"))} for t in tasks],
    })
    return {"plan_updates": updates or []}


def route_after_check_in(state: CoachState) -> Literal[
        "plan_opportunity_queries", "analyze_requirements", "assess_readiness"]:
    """Discover when opportunities are stale or --refresh; analyze only --add-posting URLs; else go on."""
    if state.get("discovery_due"):
        return "plan_opportunity_queries"
    if state.get("extra_postings"):
        return "analyze_requirements"
    return "assess_readiness"


def plan_opportunity_queries(state: CoachState) -> dict:
    """LLM (OpportunityQueries) writes NUM_QUERIES search queries covering every target type.

    Returns: {"queries": [{"type", "query"}, ...]}
    """
    raise NotImplementedError


def discover_opportunities(state: CoachState) -> dict:
    """Run tools.search_opportunities per query over the sources for its type (SEARCH_DAYS window).

    Drops URLs already in career.db and duplicates. A failing search is logged and skipped.
    Returns: {"candidates": [...]}
    """
    raise NotImplementedError


def analyze_requirements(state: CoachState) -> dict:
    """Load each candidate and --add-posting URL; LLM (PostingAnalysis) extracts requirements.

    Drops non-postings, closed postings (tools.posting_is_open), and ineligible ones; maps skills
    to skills.yaml ids; computes fit_score (target priority x skill overlap). Merges with the stored
    opportunities and keeps them ranked by fit.
    Returns: {"opportunities": [...]}
    """
    raise NotImplementedError


def assess_readiness(state: CoachState) -> dict:
    """Skill demand and targets from the opportunities; readiness from career.db history (§7).

    Returns: {"skill_demand": ..., "skill_targets": ..., "readiness": ...}
    """
    raise NotImplementedError


def plan_session(state: CoachState) -> dict:
    """Pick `session_size` slots (default SESSION_QUESTIONS): largest gaps first, one retest of a
    previously weak skill, at least one of each REQUIRED_KINDS; difficulty from difficulty_for().

    Returns: {"session_plan": [{"skill", "kind", "difficulty"}, ...]} ([] when not interactive).
    """
    raise NotImplementedError


def route_after_plan(state: CoachState) -> Literal["generate_question", "summarize_strengths_gaps"]:
    """--report-only (or non-interactive) sessions have no questions."""
    return "generate_question" if state.get("session_plan") else "summarize_strengths_gaps"


def generate_question(state: CoachState) -> dict:
    """LLM (GeneratedQuestion) writes the next slot's question, grounded in a posting requirement or
    an Agent 1 article when one fits.

    Coding: run the tests against reference_solution first and drop failing tests; write the starter
    code to WORKSPACE_DIR/<id>.py here (not in ask_and_collect_answer, which re-runs on resume).
    Regenerates once if the question is within REPEAT_SIMILARITY of a past one.
    Returns: {"questions": questions + [new]}
    """
    raise NotImplementedError


def ask_and_collect_answer(state: CoachState) -> dict:
    """Pause for my answer to the latest question (human-in-the-loop).

    Resume value: {"action": "answer" | "skip" | "quit", "answer": str, "hints_used": int}.
    """
    questions = state.get("questions", [])
    reply = interrupt({
        "kind": "question",
        "number": len(top_level_questions(questions)),
        "total": len(state.get("session_plan", [])),
        "question": public_question(questions[-1]),
    })
    return {"pending_answer": reply}


def grade_answer(state: CoachState) -> dict:
    """Grade `pending_answer` against the latest question.

    quit -> {"quit": True} with no assessment. skip -> an Assessment with action "skipped" and no
    score (not used for readiness). Otherwise LLM (Grade) scores the rubric; coding answers also run
    tools.run_tests on the workspace file. answer_score() computes the 0-1 score.
    Returns: {"assessments": assessments + [new]} or {"quit": True}
    """
    raise NotImplementedError


def route_after_grading(state: CoachState) -> Literal[
        "generate_follow_up", "generate_question", "summarize_strengths_gaps"]:
    """One follow-up per question when the answer was weak or very strong; then the next slot."""
    if state.get("quit"):
        return "summarize_strengths_gaps"
    questions, assessments = state.get("questions", []), state.get("assessments", [])
    last = assessments[-1] if assessments else None
    if (last and last.get("action") == "answered"
            and follow_ups_of(questions, questions[-1]) < config.MAX_FOLLOW_UPS
            and follow_up_mode(last["score"])):
        return "generate_follow_up"
    if len(top_level_questions(questions)) < len(state.get("session_plan", [])):
        return "generate_question"
    return "summarize_strengths_gaps"


def generate_follow_up(state: CoachState) -> dict:
    """LLM (GeneratedQuestion) with FOLLOW_UP_MODES[follow_up_mode(score)]; parent_id = the top-level id.

    Returns: {"questions": questions + [follow_up]}
    """
    raise NotImplementedError


def summarize_strengths_gaps(state: CoachState) -> dict:
    """Update readiness with today's scores in code; LLM (StrengthsGaps) writes cited findings.

    Findings whose evidence ids don't exist are dropped.
    Returns: {"readiness_after": ..., "strengths": [...], "gaps": [...]}
    """
    raise NotImplementedError


def recommend_resources_and_plan(state: CoachState) -> dict:
    """Search resources for the top PLAN_GAPS gaps (tools.search_resources + arXiv), keep those whose
    links resolve, then LLM (PrepPlanOutput) picks resources by number and writes PLAN_DAYS of tasks.
    Open tasks from the check-in are carried over.

    Returns: {"resources": [...], "prep_plan": [...]}
    """
    raise NotImplementedError


def compose_report(state: CoachState) -> dict:
    """Render the report and transcript (ARCHITECTURE.md §8) to REPORTS_DIR and SESSIONS_DIR.

    Titles, URLs, deadlines, and resource links come from stored fields, never LLM text.
    Returns: {"report_markdown", "report_path", "transcript_markdown", "transcript_path"}
    """
    raise NotImplementedError


def update_memory(state: CoachState) -> dict:
    """Write opportunities, questions, assessments, readiness, check-in updates, and the plan to
    career.db; add question embeddings to the vector store."""
    raise NotImplementedError
