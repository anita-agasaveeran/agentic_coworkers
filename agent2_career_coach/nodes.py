"""Graph nodes for nextLevelPersonalCoachForCareerLevel (one function per node in ARCHITECTURE.md §4).

Each node takes the current CoachState and returns a partial state update.
"""

import ast
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Literal

from langgraph.types import interrupt

import config
import memory
import prompts
import tools
from state import (Assessment, CoachState, GeneratedQuestion, Grade, Opportunity, OpportunityQueries,
                   PlanTask, PostingAnalysis, PrepPlanOutput, Question, Resource, SkillReq, StrengthsGaps)

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


def profile_for_prompt(profile: dict) -> str:
    """Profile JSON for prompts, without notes and derived fields; the resume is appended if present."""
    shown = {k: v for k, v in profile.items() if not k.startswith("_") and k not in ("resume", "reading")}
    text = json.dumps(shown, indent=2)
    return text + (f"\n\nResume:\n{profile['resume']}" if profile.get("resume") else "")


def skill_label(skill: str, skills: dict) -> str:
    """"rag" -> "rag (retrieval-augmented generation, ...)" for prompts and reports."""
    for area in skills["areas"].values():
        if skill in area:
            aliases = area[skill] or []
            return f"{skill.replace('_', ' ')} ({', '.join(aliases[:3])})" if aliases else skill.replace("_", " ")
    return skill.removeprefix("other:")


def skill_area(skill: str, skills: dict) -> str | None:
    return next((name for name, area in skills["areas"].items() if skill in area), None)


def taxonomy_for_prompt(skills: dict) -> str:
    return "\n".join(f"{skill}: {', '.join(aliases or [])}"
                     for area in skills["areas"].values() for skill, aliases in area.items())


def normalize_skill(raw: str, index: dict[str, str]) -> str:
    """Taxonomy id for an extracted skill name, else "other:<name>"."""
    name = " ".join(raw.lower().split()).removeprefix("other:").strip()
    return index.get(name) or index.get(name.replace(" ", "_")) or f"other:{name}"


def parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value.strip()[:10]) if value else None
    except ValueError:
        return None


def known_level(level: str | None) -> float:
    """How much a self-rated skill counts toward fit: intermediate or better 1, beginner 0.5, none 0."""
    value = self_rated(level)
    return 0.0 if value is None else 1.0 if value >= config.SELF_RATED_READINESS["intermediate"] else 0.5


def fit_score(requirements: list[SkillReq], opportunity_type: str, profile: dict) -> float:
    """Target priority x (FIT_FLOOR + (1 - FIT_FLOOR) x overlap of the requirements with my skills)."""
    priority = max((t.get("priority", 0) for t in profile.get("targets", []) if t["type"] == opportunity_type),
                   default=0.0)
    weights = [(config.REQUIRED_WEIGHT[r["required"]], known_level(profile.get("skills", {}).get(r["skill"])))
               for r in requirements]
    total = sum(w for w, _ in weights)
    overlap = sum(w * k for w, k in weights) / total if total else 0.0
    return round(priority * (config.FIT_FLOOR + (1 - config.FIT_FLOOR) * overlap), 3)


def merge_requirements(requirements: list[SkillReq]) -> list[SkillReq]:
    """One requirement per skill: highest level, required if any mention is required."""
    merged: dict[str, SkillReq] = {}
    for r in requirements:
        if r["skill"] in merged:
            m = merged[r["skill"]]
            merged[r["skill"]] = {**m, "level": max(m["level"], r["level"]), "required": m["required"] or r["required"]}
        else:
            merged[r["skill"]] = r
    return list(merged.values())


def text_contains(text: str, phrase: str | None) -> bool:
    return bool(phrase) and " ".join(phrase.lower().split()) in " ".join(text.lower().split())


def grounding_material(skill: str, opportunities: list[Opportunity], reading: dict) -> list[dict]:
    """Numbered material a question can be based on: posting requirements, then articles I read."""
    items = []
    for o in opportunities[:config.MAX_OPPORTUNITIES]:
        for r in o.get("requirements", []):
            if r["skill"] == skill and r.get("evidence") and len(items) < 4:
                items.append({"url": o["url"],
                              "text": f'Posting "{o["title"]}" ({o.get("org") or o.get("source")}) asks: "{r["evidence"]}"'})
    for article in reading.get(skill, {}).get("recent", []):
        items.append({"url": article["url"], "text": f'Article I read via Agent 1: "{article["title"]}"'})
    return items


def repeat_problem(prompt: str, recent: list[str], store) -> str | None:
    """Why a generated question can't be used as new, or None."""
    if " ".join(prompt.lower().split()) in {" ".join(r.lower().split()) for r in recent}:
        return "it repeats a past question word for word"
    if store is not None and memory.max_similarity(store, prompt) >= config.REPEAT_SIMILARITY:
        return "it is nearly the same as a past question"
    return None


def drop_tests(tests: str, names: set[str]) -> str:
    """Remove the named test functions (also inside test classes)."""
    tree = ast.parse(tests)

    def keep(node):
        return not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names)

    tree.body = [n for n in tree.body if keep(n)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.body = [n for n in node.body if keep(n)] or [ast.Pass()]
    return ast.unparse(tree)


def validated_tests(out: GeneratedQuestion) -> tuple[str | None, str | None]:
    """(tests the reference solution passes, None) or (None, why the coding question is unusable)."""
    if not out.tests or not out.reference_solution:
        return None, "it had no tests or no reference solution"
    try:
        ast.parse(out.tests)
    except SyntaxError:
        return None, "its tests were not valid Python"
    result = tools.run_tests(out.reference_solution, out.tests)
    if result["passed"] == 0:
        return None, "its reference solution passed none of its tests"
    failing = {name.split("[")[0] for name, ok in result["results"].items() if not ok}
    if failing:
        log.info("dropping tests the reference solution fails: %s", sorted(failing))
        return drop_tests(out.tests, failing), None
    return out.tests, None


def clean_list(items: list[str], limit: int) -> list[str]:
    return [i.strip() for i in items if i and i.strip()][:limit]


# --- Scoring (ARCHITECTURE.md §7) --------------------------------------------

def criterion_value(score: int) -> float:
    """1-5 rubric score -> 0-1."""
    return (min(max(int(score), 1), 5) - 1) / 4


def answer_score(kind: str, criteria: dict[str, int], tests_passed: float | None, hints_used: int) -> float:
    """0-1 answer score: mean criteria (coding: CODING_WEIGHTS with test pass rate), minus HINT_PENALTY per hint."""
    review = sum(map(criterion_value, criteria.values())) / len(criteria) if criteria else 0.0
    if kind == "coding" and tests_passed is not None:
        w = config.CODING_WEIGHTS
        review = w["tests"] * tests_passed + w["review"] * review
    return round(max(0.0, min(1.0, review - config.HINT_PENALTY * hints_used)), 3)


def self_rated(level: str | None) -> float | None:
    """Readiness for a self-rated level ("beginner", "intermediate", ...), matched on its first word."""
    words = (level or "").lower().split()
    return config.SELF_RATED_READINESS.get(words[0]) if words else None


def readiness_from_history(history: list[dict], self_rated_levels: dict[str, str], reading: dict[str, dict],
                           today) -> tuple[dict[str, float], set[str]]:
    """Per-skill readiness and the set of skills with at least one assessment.

    Recency-weighted mean (READINESS_HALF_LIFE_DAYS) of score x DIFFICULTY_FACTOR; self-rated
    fallback via SELF_RATED_READINESS (matched on the level's first word); plus up to READING_BONUS.
    """
    sums: dict[str, list[float]] = {}
    for h in history:
        weight = 0.5 ** ((today - h["session_date"]).days / config.READINESS_HALF_LIFE_DAYS)
        value = h["score"] * config.DIFFICULTY_FACTOR[h["difficulty"]]
        acc = sums.setdefault(h["skill"], [0.0, 0.0])
        acc[0] += weight * value
        acc[1] += weight
    readiness = {skill: total / weights for skill, (total, weights) in sums.items() if weights}
    assessed = set(readiness)
    for skill, level in self_rated_levels.items():
        if skill not in readiness and (value := self_rated(level)) is not None:
            readiness[skill] = value
    for skill, r in reading.items():
        bonus = config.READING_BONUS * min(r["articles"] / config.READING_FULL_AT, 1.0)
        readiness[skill] = readiness.get(skill, 0.0) + bonus
    return {skill: round(min(value, 1.0), 3) for skill, value in readiness.items()}, assessed


def skill_demand(opportunities: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """(demand, target level) per skill across the top MAX_OPPORTUNITIES by fit, demand normalized to max 1."""
    top = sorted(opportunities, key=lambda o: o.get("fit_score") or 0, reverse=True)[:config.MAX_OPPORTUNITIES]
    demand: dict[str, float] = {}
    targets: dict[str, int] = {}
    for o in top:
        for r in o.get("requirements", []):
            demand[r["skill"]] = demand.get(r["skill"], 0.0) + config.REQUIRED_WEIGHT[r["required"]] * (o.get("fit_score") or 0)
            targets[r["skill"]] = max(targets.get(r["skill"], 1), r["level"])
    peak = max(demand.values(), default=0.0)
    if peak <= 0:
        return {}, {}
    demand = {skill: round(value / peak, 3) for skill, value in sorted(demand.items(), key=lambda kv: -kv[1])}
    return demand, {skill: targets[skill] for skill in demand}


def skill_gaps(demand: dict[str, float], targets: dict[str, int], readiness: dict[str, float]) -> dict[str, float]:
    """gap = demand x max(0, TARGET_READINESS[target] - readiness), largest first.

    Skills outside the taxonomy ("other:...") are left out: they can't be practiced or tracked.
    """
    gaps = {}
    for skill, d in demand.items():
        if skill.startswith("other:"):
            continue
        target = config.TARGET_READINESS[targets.get(skill, 2)]
        gaps[skill] = round(d * max(0.0, target - readiness.get(skill, 0.0)), 3)
    return dict(sorted(gaps.items(), key=lambda kv: -kv[1]))


def role_readiness(target: dict, opportunities: list[dict], readiness: dict[str, float]) -> dict:
    """{"readiness": 0-1 | None, "boost": [three skills that would raise it most]} for one profile target.

    Demand-weighted mean of min(readiness / target readiness, 1) over the skills that stored
    postings of the target's type ask for. None when there are no such postings.
    """
    demand, targets = skill_demand([o for o in opportunities if o.get("type") == target["type"]])
    demand = {s: d for s, d in demand.items() if not s.startswith("other:")}
    if not demand:
        return {"readiness": None, "boost": []}
    met = {s: min(readiness.get(s, 0.0) / config.TARGET_READINESS[targets[s]], 1.0) for s in demand}
    total = sum(demand.values())
    shortfall = sorted(demand, key=lambda s: -demand[s] * (1 - met[s]))
    return {"readiness": round(sum(demand[s] * met[s] for s in demand) / total, 3),
            "boost": [s for s in shortfall if met[s] < 1][:3]}


def difficulty_for(readiness: float) -> int:
    """Question difficulty 1-3 from DIFFICULTY_BANDS."""
    return next(level for bound, level in config.DIFFICULTY_BANDS if readiness < bound)


# --- Nodes -------------------------------------------------------------------

def load_profile(state: CoachState) -> dict:
    """Load profile.json, resume.md, stored opportunities, open plan tasks, and Agent 1 reading.

    Discovery is due with --refresh, with no stored opportunities, or when the newest is older
    than OPPORTUNITY_MAX_AGE_DAYS, and only when a Tavily key is set.
    """
    session_date = state.get("session_date") or date.today()
    profile = {**config.load_profile_file(), "resume": config.load_resume(),
               "reading": memory.agent1_reading(config.load_skills())}
    conn = memory.connect()
    try:
        opportunities = memory.stored_opportunities(conn, session_date)
        newest = memory.newest_opportunity_date(conn)
        open_tasks = memory.open_plan_tasks(conn)
    finally:
        conn.close()
    stale = newest is None or (session_date - newest).days > config.OPPORTUNITY_MAX_AGE_DAYS
    due = bool(state.get("refresh") or not opportunities or stale)
    if due and not os.getenv("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set: skipping opportunity discovery, using stored opportunities")
        due = False
    return {
        "session_date": session_date,
        "session_size": state.get("session_size", config.SESSION_QUESTIONS),
        "profile": profile,
        "opportunities": opportunities,
        "open_tasks": open_tasks,
        "discovery_due": due,
        "plan_updates": [], "queries": [], "candidates": [], "rejected": [], "posting_texts": {},
        "questions": [], "assessments": [], "pending_answer": None, "quit": False,
    }


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
    """LLM (OpportunityQueries) writes NUM_QUERIES search queries covering every target type."""
    types = sorted({t["type"] for t in state["profile"].get("targets", [])})
    lo, hi = config.NUM_QUERIES
    plan = structured(OpportunityQueries).invoke(prompts.PLAN_OPPORTUNITY_QUERIES.format(
        profile=profile_for_prompt(state["profile"]), types=", ".join(types), min_queries=lo, max_queries=hi))
    queries, seen = [], set()
    for q in plan.queries:
        text = " ".join(q.query.split())
        if q.type in types and text and text.lower() not in seen:
            seen.add(text.lower())
            queries.append({"type": q.type, "query": text})
    return {"queries": queries[:hi]}


def discover_opportunities(state: CoachState) -> dict:
    """Run tools.search_opportunities for each query and its type (SEARCH_DAYS window).

    Drops URLs already in career.db and duplicates, and takes results round-robin across
    queries up to MAX_CANDIDATES so every type is represented. A failing search is skipped.
    """
    sources = config.load_opportunity_sources()["opportunity_sources"]
    conn = memory.connect()
    try:
        seen = memory.known_urls(conn)
    finally:
        conn.close()
    per_query = []
    for q in state.get("queries", []):
        try:
            per_query.append(tools.search_opportunities(q["query"], q["type"], sources, config.SEARCH_DAYS,
                                                        config.MAX_RESULTS_PER_QUERY))
        except Exception as exc:
            log.warning("opportunity search failed: %s (%s)", q["query"], exc)
    candidates = []
    for found in itertools.chain.from_iterable(itertools.zip_longest(*per_query)):
        if found and found["url"] not in seen:
            seen.add(found["url"])
            candidates.append({**found, "found_on": state["session_date"]})
    log.info("discovered %d new postings", len(candidates))
    return {"candidates": candidates[:config.MAX_CANDIDATES]}


def analyze_requirements(state: CoachState) -> dict:
    """Load each candidate and --add-posting URL (tools.load_posting); LLM (PostingAnalysis)
    extracts requirements.

    Drops non-postings, closed postings (tools.posting_is_open), and ineligible ones (kept in
    `rejected` so they aren't analyzed again); maps skills to skills.yaml ids; computes fit_score.
    Merges with the stored opportunities, ranked by fit. Titles come from the search result; for
    --add-posting URLs, from the LLM only if the title appears in the page text.
    """
    today = state["session_date"]
    stored = state.get("opportunities", [])
    items = list(state.get("candidates", []))
    taken = {o["url"] for o in stored} | {c["url"] for c in items}
    for url in state.get("extra_postings", []):
        url = tools.canonical_url(url)
        if url not in taken:
            taken.add(url)
            items.append({"url": url, "title": None, "org": None, "type": None, "source": "Added by me",
                          "found_on": today, "snippet": ""})
    if not items:
        return {"posting_texts": {}}

    with ThreadPoolExecutor(max_workers=8) as pool:
        postings = list(pool.map(tools.load_posting, [i["url"] for i in items]))
    rejected, readable = [], []
    for item, posting in zip(items, postings):
        if posting is None:
            log.warning("couldn't read posting: %s", item["url"])
            if item.get("title"):
                rejected.append({**item, "status": "dismissed", "requirements": []})
        else:
            readable.append((item, posting))

    skills = config.load_skills()
    index = memory.alias_index(skills)
    profile = state["profile"]
    requests_ = [prompts.ANALYZE_POSTING.format(
        profile=profile_for_prompt(profile), taxonomy=taxonomy_for_prompt(skills),
        title=item.get("title") or "(unknown)", url=item["url"], text=posting["text"][:config.MAX_POSTING_CHARS],
    ) for item, posting in readable]
    analyses = structured(PostingAnalysis).batch(requests_, return_exceptions=True)

    new, texts = [], {}
    for (item, posting), analysis in zip(readable, analyses):
        if isinstance(analysis, Exception):
            log.warning("posting analysis failed: %s (%s)", item["url"], analysis)
            continue
        title = item.get("title") or (analysis.title if text_contains(posting["text"], analysis.title) else item["url"])
        deadline = parse_date(analysis.deadline)
        opportunity = Opportunity(
            url=item["url"], title=title, type=item.get("type") or analysis.type, source=item.get("source"),
            snippet=item.get("snippet", ""), org=item.get("org") or analysis.org, location=analysis.location,
            deadline=deadline, found_on=item["found_on"], requirements=[], fit_score=0.0, status="open",
        )
        if not analysis.is_posting or not analysis.eligible:
            rejected.append({**opportunity, "status": "dismissed"})
            continue
        if analysis.closed or not tools.posting_is_open(posting, deadline, today):
            rejected.append({**opportunity, "status": "closed"})
            continue
        requirements = merge_requirements([SkillReq(
            skill=normalize_skill(r.skill, index), level=min(max(int(r.level), 1), 3),
            required=bool(r.required), evidence=" ".join(r.evidence.split()),
        ) for r in analysis.requirements if r.skill.strip()])
        new.append({**opportunity, "requirements": requirements,
                    "fit_score": fit_score(requirements, opportunity["type"], profile)})
        texts[item["url"]] = posting["text"][:config.MAX_POSTING_CHARS]
    log.info("analyzed %d postings: %d kept, %d dropped", len(items), len(new), len(rejected))
    ranked = sorted(stored + new, key=lambda o: o.get("fit_score") or 0, reverse=True)
    return {"opportunities": ranked, "rejected": rejected, "posting_texts": texts}


def assess_readiness(state: CoachState) -> dict:
    """Skill demand and targets from the opportunities; readiness from career.db history (§7).

    With no analyzed postings yet, demand falls back to the skills in my profile (target level 2)
    so sessions still work.
    """
    today = state["session_date"]
    profile = state["profile"]
    conn = memory.connect()
    try:
        history = memory.assessment_history(conn, before=today)
        previous = memory.previous_readiness(conn, today)
    finally:
        conn.close()
    readiness, assessed = readiness_from_history(history, profile.get("skills", {}), profile.get("reading", {}), today)
    demand, targets = skill_demand(state.get("opportunities", []))
    if not demand:
        log.warning("no analyzed postings: using the skills in profile.json as demand")
        demand = {skill: 1.0 for skill in profile.get("skills", {})}
        targets = {skill: 2 for skill in demand}
    latest: dict[str, tuple[date, list[float]]] = {}
    for h in history:    # oldest first: keep the scores from each skill's latest session
        day, scores = latest.get(h["skill"], (None, []))
        latest[h["skill"]] = (h["session_date"], scores + [h["score"]] if day == h["session_date"] else [h["score"]])
    mean = {s: sum(scores) / len(scores) for s, (_, scores) in latest.items()}
    weak = sorted((s for s in mean if mean[s] < config.PROBE_BELOW), key=lambda s: (latest[s][0], -mean[s]),
                  reverse=True)
    return {"skill_demand": demand, "skill_targets": targets, "readiness": readiness,
            "assessed_skills": sorted(assessed), "previous": previous, "weak_skills": weak}


def plan_session(state: CoachState) -> dict:
    """Pick `session_size` slots (default SESSION_QUESTIONS): largest gaps first, one retest of a
    previously weak skill, at least one of each REQUIRED_KINDS; difficulty from difficulty_for().

    Returns: {"session_plan": [{"skill", "kind", "difficulty", "reason"}, ...]} ([] when not interactive).
    """
    n = state.get("session_size", config.SESSION_QUESTIONS)
    if not state.get("interactive", True) or n <= 0:
        return {"session_plan": []}
    readiness = state.get("readiness", {})
    demand = state.get("skill_demand", {})
    gaps = skill_gaps(demand, state.get("skill_targets", {}), readiness)
    ranked = ([s for s, g in gaps.items() if g > 0] or [s for s in demand if not s.startswith("other:")]
              or list(state["profile"].get("skills", {})) or ["python"])

    retest = next((s for s in state.get("weak_skills", []) if s not in ranked[:n - 1]), None) if n > 1 else None
    picks = [(s, "gap") for s in ranked[:n - (1 if retest else 0)]]
    if retest:
        picks.append((retest, "retest"))
    while len(picks) < n:   # fewer gap skills than slots: practice them again
        picks.append((ranked[len(picks) % len(ranked)], "gap"))

    if "coding" in config.REQUIRED_KINDS and n >= len(config.REQUIRED_KINDS) \
            and not any(s in config.CODING_SKILLS for s, _ in picks):
        codable = next((s for s in [*ranked, *demand, "python"] if s in config.CODING_SKILLS), "python")
        last_gap = max(i for i, (_, reason) in enumerate(picks) if reason == "gap")
        picks[last_gap] = (codable, "gap")

    kinds = ["concept"] * n
    for kind in config.REQUIRED_KINDS:
        free = [i for i in range(n) if kinds[i] == "concept"]
        if kind == "coding":
            slot = next((i for i in free if picks[i][0] in config.CODING_SKILLS), None)
        else:
            slot = next((i for i in free if picks[i][0] not in config.CODING_SKILLS), free[0] if free else None)
        if slot is not None:
            kinds[slot] = kind
    return {"session_plan": [{"skill": s, "kind": k, "difficulty": difficulty_for(readiness.get(s, 0.0)), "reason": r}
                             for (s, r), k in zip(picks, kinds)]}


def route_after_plan(state: CoachState) -> Literal["generate_question", "summarize_strengths_gaps"]:
    """--report-only (or non-interactive) sessions have no questions."""
    return "generate_question" if state.get("session_plan") else "summarize_strengths_gaps"


def generate_question(state: CoachState) -> dict:
    """LLM (GeneratedQuestion) writes the next slot's question, grounded in a posting requirement or
    an Agent 1 article when one fits.

    Coding: the tests run against reference_solution first and failing tests are dropped; after two
    unusable coding drafts the slot becomes a concept question. A draft that repeats a past question
    is regenerated once. The starter code is written to WORKSPACE_DIR/<id>.py here (not in
    ask_and_collect_answer, which re-runs on resume).
    """
    questions = state.get("questions", [])
    number = len(top_level_questions(questions)) + 1
    slot = state["session_plan"][number - 1]
    qid = f"{state['session_date'].isoformat()}-q{number}"
    skills = config.load_skills()
    material = grounding_material(slot["skill"], state.get("opportunities", []),
                                  state["profile"].get("reading", {}))
    conn = memory.connect()
    try:
        recent = memory.recent_questions(conn, slot["skill"])
    finally:
        conn.close()
    recent += [q["prompt"] for q in questions if q["skill"] == slot["skill"]]
    embeddings = embeddings_or_none()
    store = memory.load_vector_store(embeddings) if embeddings is not None else None

    kind, note, tests = slot["kind"], "", None
    for attempt in range(3):
        if attempt == 2 and kind == "coding":
            log.warning("no usable coding question for %s; asking a concept question instead", slot["skill"])
            kind = "concept"
        out = structured(GeneratedQuestion).invoke(prompts.GENERATE_QUESTION.format(
            kind=kind, skill=slot["skill"], skill_aliases=skill_label(slot["skill"], skills),
            difficulty=slot["difficulty"], profile=profile_for_prompt(state["profile"]),
            grounding="\n".join(f"{i}. {m['text']}" for i, m in enumerate(material, 1)) or "(none)",
            recent_questions="\n".join(f"- {r}" for r in recent[-10:]) or "(none)",
            kind_instructions=prompts.KIND_INSTRUCTIONS[kind], hints=config.HINTS_PER_QUESTION,
        ) + note)
        problem = repeat_problem(out.prompt, recent, store)
        if problem is None and kind == "coding":
            tests, problem = validated_tests(out)
        if problem is None or (attempt >= 1 and kind != "coding"):
            break
        log.info("regenerating question %s: %s", qid, problem)
        note = f"\n\nA previous draft was rejected because {problem}. Write a different question."

    grounding = material[out.grounding_id - 1]["url"] if out.grounding_id and 0 < out.grounding_id <= len(material) else None
    question = Question(
        id=qid, skill=slot["skill"], kind=kind, difficulty=slot["difficulty"], prompt=out.prompt.strip(),
        grounding=grounding, reference_answer=out.reference_answer.strip(),
        rubric=clean_list(out.rubric, 4) or ["correctness", "completeness", "clarity"],
        hints=clean_list(out.hints, config.HINTS_PER_QUESTION),
        starter_code=None, tests=None, reference_solution=None, workspace_file=None, parent_id=None,
    )
    if kind == "coding":
        config.WORKSPACE_DIR.mkdir(exist_ok=True)
        path = config.WORKSPACE_DIR / f"{qid}.py"
        path.write_text(out.starter_code or "# Write your solution here.\n")
        question.update(starter_code=out.starter_code, tests=tests, reference_solution=out.reference_solution,
                        workspace_file=str(path))
    return {"questions": questions + [question]}


def ask_and_collect_answer(state: CoachState) -> dict:
    """Pause for my answer to the latest question (human-in-the-loop).

    The payload also carries the grade of the previous answer, so I see it before the next question.
    Resume value: {"action": "answer" | "skip" | "quit", "answer": str, "hints_used": int}.
    """
    questions = state.get("questions", [])
    assessments = state.get("assessments", [])
    last = assessments[-1] if assessments else None
    previous = None
    if last and last["question_id"] != questions[-1]["id"]:
        previous = {k: last.get(k) for k in
                    ("question_id", "action", "score", "tests_passed", "feedback", "misconceptions")}
    reply = interrupt({
        "kind": "question",
        "number": len(top_level_questions(questions)),
        "total": len(state.get("session_plan", [])),
        "question": public_question(questions[-1]),
        "previous": previous,
    })
    return {"pending_answer": reply}


def match_criteria(rubric: list[str], grade: Grade) -> dict[str, int]:
    """Rubric criterion -> 1-5, matched by name, else by position."""
    by_name = {" ".join(c.criterion.lower().split()): c.score for c in grade.criteria}
    scores = {}
    for i, criterion in enumerate(rubric):
        score = by_name.get(" ".join(criterion.lower().split()))
        if score is None and i < len(grade.criteria):
            score = grade.criteria[i].score
        if score is not None:
            scores[criterion] = min(max(int(score), 1), 5)
    return scores


def grade(question: Question, answer: str, test_text: str = "") -> tuple[dict[str, int], Grade]:
    """LLM (Grade) rubric scores for one answer; shared with evaluate.py's grader check."""
    out = structured(Grade).invoke(prompts.GRADE_ANSWER.format(
        kind=question["kind"], skill=question["skill"], difficulty=question["difficulty"],
        question=question["prompt"], reference_answer=question["reference_answer"],
        rubric="\n".join(f"- {c}" for c in question["rubric"]), answer=answer, test_results=test_text,
    ))
    return match_criteria(question["rubric"], out), out


def grade_answer(state: CoachState) -> dict:
    """Grade `pending_answer` against the latest question.

    quit -> {"quit": True} with no assessment. skip -> an Assessment with action "skipped" and no
    score (not used for readiness). Otherwise LLM (Grade) scores the rubric; coding answers also run
    tools.run_tests on the code read from the workspace file. answer_score() computes the 0-1 score.
    """
    reply = state.get("pending_answer") or {}
    question = state["questions"][-1]
    assessments = state.get("assessments", [])
    if reply.get("action") == "quit":
        return {"quit": True}
    hints_used = int(reply.get("hints_used", 0))
    if reply.get("action") == "skip":
        return {"assessments": assessments + [Assessment(
            question_id=question["id"], action="skipped", answer="", hints_used=hints_used, criteria_scores={},
            tests_passed=None, score=None, feedback="Skipped.", misconceptions=[])]}

    answer = reply.get("answer", "")
    tests_passed, test_text = None, ""
    if question["kind"] == "coding" and question.get("tests"):
        result = tools.run_tests(answer, question["tests"])
        tests_passed = result["pass_rate"]
        status = " (timed out)" if result["timed_out"] else ""
        test_text = (f"Test results{status}: {result['passed']}/{result['total']} passed.\n"
                     f"{result['output'][-1500:]}")
    criteria, out = grade(question, answer, test_text)
    return {"assessments": assessments + [Assessment(
        question_id=question["id"], action="answered", answer=answer, hints_used=hints_used,
        criteria_scores=criteria, tests_passed=tests_passed,
        score=answer_score(question["kind"], criteria, tests_passed, hints_used),
        feedback=out.feedback.strip(), misconceptions=clean_list(out.misconceptions, 5),
    )]}


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

    Follow-ups are answered in text (a coding question's follow-up is a concept question); a
    challenge is one level harder.
    """
    questions, assessments = state["questions"], state["assessments"]
    parent, last = questions[-1], assessments[-1]
    root = parent.get("parent_id") or parent["id"]
    mode = follow_up_mode(last["score"])
    out = structured(GeneratedQuestion).invoke(prompts.FOLLOW_UP.format(
        question=parent["prompt"], answer=last["answer"][:4000], score=last["score"],
        misconceptions="; ".join(last.get("misconceptions", [])) or "none",
        mode_instructions=prompts.FOLLOW_UP_MODES[mode], hints=config.HINTS_PER_QUESTION,
    ))
    follow_up = Question(
        id=f"{root}f", skill=parent["skill"], kind="scenario" if parent["kind"] == "scenario" else "concept",
        difficulty=min(parent["difficulty"] + 1, 3) if mode == "challenge" else parent["difficulty"],
        prompt=out.prompt.strip(), grounding=parent.get("grounding"), reference_answer=out.reference_answer.strip(),
        rubric=clean_list(out.rubric, 3) or ["correctness", "reasoning"],
        hints=clean_list(out.hints, config.HINTS_PER_QUESTION),
        starter_code=None, tests=None, reference_solution=None, workspace_file=None, parent_id=root,
    )
    return {"questions": questions + [follow_up]}


def requirement_refs(opportunities: list[Opportunity], demand: dict[str, float], limit: int = 40) -> list[tuple[str, str]]:
    """("R<n>", description) for posting requirements on taxonomy skills, highest-demand skills first."""
    reqs = [(o, r) for o in opportunities[:config.MAX_OPPORTUNITIES] for r in o.get("requirements", [])
            if not r["skill"].startswith("other:")]
    reqs.sort(key=lambda pair: -demand.get(pair[1]["skill"], 0))
    return [(f"R{i}", f'{r["skill"]} (level {r["level"]}, {"required" if r["required"] else "preferred"}) '
                      f'in "{o["title"]}" ({o.get("org") or o.get("source")}): "{r["evidence"]}"')
            for i, (o, r) in enumerate(reqs[:limit], 1)]


def summarize_strengths_gaps(state: CoachState) -> dict:
    """Update readiness with today's scores in code; LLM (StrengthsGaps) writes cited findings.

    Evidence ids that don't exist are removed, and findings left without evidence are dropped.
    """
    today = state["session_date"]
    profile = state["profile"]
    questions = {q["id"]: q for q in state.get("questions", [])}
    answered = [a for a in state.get("assessments", []) if a["action"] == "answered"]
    conn = memory.connect()
    try:
        history = memory.assessment_history(conn, before=today)
    finally:
        conn.close()
    todays = [{"skill": questions[a["question_id"]]["skill"], "difficulty": questions[a["question_id"]]["difficulty"],
               "score": a["score"], "session_date": today} for a in answered]
    readiness, assessed = readiness_from_history(history + todays, profile.get("skills", {}),
                                                 profile.get("reading", {}), today)
    demand, targets = state.get("skill_demand", {}), state.get("skill_targets", {})
    result = {"readiness_after": readiness, "assessed_after": sorted(assessed), "strengths": [], "gaps": []}
    if not demand and not answered:
        return result

    refs = requirement_refs(state.get("opportunities", []), demand)
    shown = [s for s in demand if not s.startswith("other:")][:15]
    shown += [q["skill"] for q in questions.values() if q["skill"] not in shown]
    out = structured(StrengthsGaps).invoke(prompts.STRENGTHS_GAPS.format(
        targets="\n".join(f"- {t['role']} ({t['type']}, {t.get('timeline') or 'no timeline'})"
                          for t in profile.get("targets", [])),
        demand="\n".join(f"- {s}: {demand.get(s, 0):.2f}, level {targets.get(s, 2)}" for s in shown)
               + "\n\nPosting requirements:\n" + "\n".join(f"{rid}: {text}" for rid, text in refs),
        readiness="\n".join(f"- {s}: {readiness.get(s, 0.0):.2f}" + ("" if s in assessed else " (not yet assessed)")
                            for s in shown),
        assessments="\n".join(
            f"- {a['question_id']} | {questions[a['question_id']]['skill']} | {questions[a['question_id']]['kind']} | "
            f"score {a['score']:.2f} | misconceptions: {'; '.join(a.get('misconceptions', [])) or 'none'}"
            for a in answered) or "(no questions answered today)",
    ))
    valid = set(questions) | {rid for rid, _ in refs}

    def cited(findings):
        kept = []
        for f in findings:
            evidence = [e.strip() for e in f.evidence if e.strip() in valid]
            if evidence:
                kept.append({"skill": f.skill.strip(), "summary": f.summary.strip(), "evidence": evidence})
        return kept

    return {**result, "strengths": cited(out.strengths), "gaps": cited(out.gaps), "requirement_refs": dict(refs)}


def resource_query(skill: str, skills: dict) -> str:
    """ "rag" -> "rag retrieval-augmented generation tutorial" (the id plus its first alias)."""
    area = skills["areas"].get(skill_area(skill, skills) or "", {})
    aliases = area.get(skill) or []
    return " ".join([skill.replace("_", " "), *aliases[:1], "tutorial"])


def find_resources(skills_to_cover: list[str], today: date) -> list[Resource]:
    """Allow-listed learning resources (and arXiv papers for research skills) whose links resolve."""
    if not os.getenv("TAVILY_API_KEY"):
        log.warning("TAVILY_API_KEY not set: preparation plan without new resources")
        return []
    sources = config.load_opportunity_sources()
    skills = config.load_skills()
    found: list[Resource] = []
    for skill in skills_to_cover:
        try:
            results = tools.search_resources(resource_query(skill, skills), sources["resource_sources"],
                                             max_results=config.RESOURCES_PER_GAP * 2)
        except Exception as exc:
            log.warning("resource search failed for %s (%s)", skill, exc)
            results = []
        kept = [r for r in results if tools.url_resolves(r["url"])][:config.RESOURCES_PER_GAP]
        if skill_area(skill, skills) in config.RESEARCH_AREAS:
            papers = tools.search_arxiv(skill.replace("_", " "), sources["arxiv_categories"],
                                        since=today - timedelta(days=config.ARXIV_RESOURCE_DAYS), max_results=2)
            kept += [{"url": p["url"], "title": p["title"], "source": "arXiv", "snippet": p.get("snippet", "")}
                     for p in papers]
        found += [Resource(url=r["url"], title=r["title"], source=r["source"], skill=skill, why=r.get("snippet", ""))
                  for r in kept]
    return found


def recommend_resources_and_plan(state: CoachState) -> dict:
    """Search resources for the top PLAN_GAPS gaps, keep those whose links resolve, then LLM
    (PrepPlanOutput) picks resources by number and writes PLAN_DAYS of tasks.

    Open tasks from the check-in are carried over as they are; new tasks are capped at
    PLAN_MAX_HOURS. Resource titles and URLs come from the search results.
    """
    today = state["session_date"]
    readiness = state.get("readiness_after") or state.get("readiness", {})
    gaps = skill_gaps(state.get("skill_demand", {}), state.get("skill_targets", {}), readiness)
    top = [s for s, g in gaps.items() if g > 0][:config.PLAN_GAPS]
    updates = {u["id"]: u["status"] for u in state.get("plan_updates", [])}
    carried = [{**t, "status": updates.get(t["id"], t["status"])} for t in state.get("open_tasks", [])]
    carried = [t for t in carried if t["status"] in ("open", "partial")]
    if not top:
        return {"resources": [], "prep_plan": carried}

    candidates = find_resources(top, today)
    questions = {q["id"]: q for q in state.get("questions", [])}
    misconceptions: dict[str, list[str]] = {}
    for a in state.get("assessments", []):
        if a["action"] == "answered":
            misconceptions.setdefault(questions[a["question_id"]]["skill"], []).extend(a.get("misconceptions", []))
    targets = state.get("skill_targets", {})
    out = structured(PrepPlanOutput).invoke(prompts.RECOMMEND_PLAN.format(
        days=config.PLAN_DAYS, max_hours=config.PLAN_MAX_HOURS,
        gaps="\n".join(f"- {s}: gap {gaps[s]:.2f}, readiness {readiness.get(s, 0.0):.2f}, target "
                       f"{config.TARGET_READINESS[targets.get(s, 2)]:.2f}; today's misconceptions: "
                       f"{'; '.join(misconceptions.get(s, [])) or 'none'}" for s in top),
        carry_over="\n".join(f"- {t['task']} ({t['skill']}, {t['status']}, due {t.get('due')})" for t in carried)
                   or "(none)",
        resources="\n".join(f"[{i}] {r['title']} ({r['source']}) for {r['skill']}: {r['why'][:200]}"
                            for i, r in enumerate(candidates, 1)) or "(none found)",
    ))

    picked: dict[int, Resource] = {}
    for pick in out.resources:
        if 0 < pick.id <= len(candidates) and pick.id not in picked:
            picked[pick.id] = {**candidates[pick.id - 1], "why": pick.why.strip()}
    tasks, hours = [], 0.0
    for t in sorted(out.tasks, key=lambda t: t.due_in_days):
        task_hours = round(min(max(float(t.hours), 0.25), 8.0), 2)
        if hours + task_hours > config.PLAN_MAX_HOURS:
            continue
        hours += task_hours
        resource = None
        if t.resource_id and 0 < t.resource_id <= len(candidates):
            resource = picked.setdefault(t.resource_id, candidates[t.resource_id - 1])
        tasks.append(PlanTask(
            task=t.task.strip(), skill=t.skill.strip(), resource_url=resource["url"] if resource else None,
            hours=task_hours, due=today + timedelta(days=min(max(int(t.due_in_days), 1), config.PLAN_DAYS)),
            done_when=t.done_when.strip(), status="open",
        ))
    return {"resources": list(picked.values()), "prep_plan": tasks + carried}


# --- Report ------------------------------------------------------------------

def fmt_date(d: date | None) -> str:
    return f"{d:%b} {d.day}, {d.year}" if d else "not stated"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{round(value * 100)}%"


def md_cell(text) -> str:
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def criteria_lines(assessment: Assessment) -> list[str]:
    return [f"- {score}/5 · {criterion}" for criterion, score in assessment.get("criteria_scores", {}).items()]


def render_report(state: CoachState, trend: dict[str, list[tuple[str, float]]]) -> str:
    """Career Readiness Report (ARCHITECTURE.md §8). Titles, URLs, and deadlines come from stored fields."""
    today = state["session_date"]
    profile = state["profile"]
    skills = config.load_skills()
    opportunities = state.get("opportunities", [])
    readiness = state.get("readiness_after") or state.get("readiness", {})
    assessed = set(state.get("assessed_after") or state.get("assessed_skills", []))
    previous = state.get("previous") or {}
    demand, targets = state.get("skill_demand", {}), state.get("skill_targets", {})
    gaps = skill_gaps(demand, targets, readiness)
    questions = {q["id"]: q for q in state.get("questions", [])}

    lines = [f"# Career Readiness Report — {fmt_date(today)}", ""]
    ranked_targets = sorted(profile.get("targets", []), key=lambda t: -t.get("priority", 0))
    if ranked_targets:
        main = ranked_targets[0]
        now = role_readiness(main, opportunities, readiness)["readiness"]
        line = f"Target: {main['role']}" + (f", {main['timeline']}" if main.get("timeline") else "")
        line += f" · Readiness {pct(now)}"
        if previous and now is not None:
            before = role_readiness(main, opportunities, previous["readiness"])["readiness"]
            if before is not None:
                line += f" ({round((now - before) * 100):+d} since {fmt_date(previous['date'])})"
        lines += [line, ""]
    if state.get("discovery_due") is False and not state.get("candidates") and opportunities:
        newest = max((o["found_on"] for o in opportunities), default=None)
        lines += [f"_Opportunities from stored searches (newest found {fmt_date(newest)})._", ""]
    if not opportunities:
        lines += ["_No opportunities analyzed yet; skill demand comes from the skills in profile.json._", ""]

    lines += ["## Readiness by target", "", "| Target | Type | Priority | Readiness | Would raise it most |",
              "|---|---|---|---|---|"]
    for t in ranked_targets:
        r = role_readiness(t, opportunities, readiness)
        lines.append(f"| {md_cell(t['role'])} | {t['type']} | {t.get('priority', 0):g} | {pct(r['readiness'])} | "
                     f"{', '.join(r['boost']) or '—'} |")
    lines.append("")

    lines += [f"## Opportunities (top {min(5, len(opportunities))} by fit)", ""]
    if opportunities:
        lines += ["| Role | Org | Type | Deadline | Fit | Link |", "|---|---|---|---|---|---|"]
        for o in opportunities[:5]:
            lines.append(f"| {md_cell(o['title'])} | {md_cell(o.get('org') or '—')} | {o['type']} | "
                         f"{fmt_date(o.get('deadline'))} | {o.get('fit_score') or 0:.2f} | <{o['url']}> |")
    else:
        lines.append("None stored yet. Run with `--refresh` (needs TAVILY_API_KEY) or `--add-posting <url>`.")
    lines.append("")

    lines += ["## What these roles ask for", "", "| Skill | Demand | Level asked | My readiness | Gap |",
              "|---|---|---|---|---|"]
    for s, d in list(demand.items())[:12]:
        if s.startswith("other:"):
            lines.append(f"| {md_cell(s.removeprefix('other:'))} (outside taxonomy) | {d:.2f} | {targets.get(s, 2)} | — | — |")
            continue
        rated = ("" if s in assessed else " (self-rated)" if s in profile.get("skills", {})
                 else " (reading only)" if s in readiness else " (no evidence)")
        lines.append(f"| {s} | {d:.2f} | {targets.get(s, 2)} | {pct(readiness.get(s, 0.0))}{rated} | {gaps[s]:.2f} |")
    lines.append("")

    lines += ["## Today's assessment", ""]
    by_question = {a["question_id"]: a for a in state.get("assessments", [])}
    if not questions:
        lines.append("No questions this session.")
    for q in questions.values():
        a = by_question.get(q["id"])
        label = "Follow-up" if q.get("parent_id") else q["id"].rsplit("-", 1)[-1].upper()
        score = "skipped" if a and a["action"] == "skipped" else f"score {a['score']:.2f}" if a else "not answered"
        lines += [f"### {label} · {q['skill']} · {q['kind']} · difficulty {q['difficulty']} · {score}", "",
                  f"**Question:** {q['prompt']}", ""]
        if a and a["action"] == "answered":
            excerpt = a["answer"].strip()
            excerpt = excerpt if len(excerpt) <= 600 else excerpt[:600] + " …"
            if q["kind"] == "coding":
                lines += [f"**My answer:** see the transcript. Tests passed: {pct(a.get('tests_passed'))}", ""]
            else:
                lines += ["**My answer (excerpt):**", "", *[f"> {line}" for line in excerpt.splitlines()], ""]
            hints = f" (hints used: {a['hints_used']})" if a.get("hints_used") else ""
            lines += [f"**Scores**{hints}:", *criteria_lines(a), "", f"**Feedback:** {a['feedback']}", ""]
            if a.get("misconceptions"):
                lines += ["**Misconceptions:**", *[f"- {m}" for m in a["misconceptions"]], ""]
    if state.get("quit"):
        lines += ["_Session ended early (quit)._", ""]

    refs = state.get("requirement_refs", {})
    for heading, findings in (("Strengths", state.get("strengths", [])), ("Gaps (largest first)", state.get("gaps", []))):
        lines += [f"## {heading}", ""]
        lines += [f"- **{f['skill']}**: {f['summary']} _(evidence: {', '.join(f['evidence'])})_" for f in findings] or ["—"]
        lines.append("")

    lines += ["## Progress", ""]
    today_iso = today.isoformat()
    rows = []
    for s in sorted(set(trend) | assessed):
        points = [p for p in trend.get(s, []) if p[0] != today_iso]
        if s in assessed:
            points.append((today_iso, readiness.get(s, 0.0)))
        if points:
            rows.append(f"| {s} | {' → '.join(pct(v) for _, v in points[-6:])} | {len(points)} |")
    if rows:
        lines += ["Assessed readiness per skill (oldest → latest session):", "",
                  "| Skill | Readiness | Sessions |", "|---|---|---|", *rows, ""]
    else:
        lines += ["No assessed skills yet.", ""]
    done = [u for u in state.get("plan_updates", []) if u["status"] == "done"]
    tasks_by_id = {t["id"]: t for t in state.get("open_tasks", [])}
    if state.get("plan_updates"):
        lines += [f"Plan tasks completed since the last session: {len(done)} of {len(state['plan_updates'])}"]
        lines += [f"- {tasks_by_id[u['id']]['task']}" for u in done if u["id"] in tasks_by_id]
        lines.append("")

    resources_by_url = {r["url"]: r for r in state.get("resources", [])}
    lines += [f"## {config.PLAN_DAYS}-day preparation plan", ""]
    plan = state.get("prep_plan", [])
    if plan:
        lines += ["| Task | Skill | Resource | Hours | Due | Done when |", "|---|---|---|---|---|---|"]
        for t in plan:
            url = t.get("resource_url")
            resource = f"[{md_cell(resources_by_url[url]['title'])}]({url})" if url in resources_by_url else (f"<{url}>" if url else "—")
            carried = " _(carried over)_" if t.get("id") is not None else ""
            lines.append(f"| {md_cell(t['task'])}{carried} | {t['skill']} | {resource} | {t.get('hours') or 0:g} | "
                         f"{fmt_date(t.get('due'))} | {md_cell(t.get('done_when'))} |")
    else:
        lines.append("No gaps above target; no new tasks.")
    lines.append("")

    references = [(o["title"], o.get("org") or o.get("source"), o["url"]) for o in opportunities[:5]]
    references += [(r["title"], r["source"], r["url"]) for r in state.get("resources", [])]
    if references:
        lines += ["### References", ""]
        lines += [f"{i}. {title.rstrip('.')}. {source.rstrip('.')}. {url}"
                  for i, (title, source, url) in enumerate(references, 1)]
    return "\n".join(lines).rstrip() + "\n"


def render_transcript(state: CoachState) -> str:
    """Full session transcript: questions, my answers, hints, follow-ups, feedback, and reference answers."""
    lines = [f"# Coaching Session — {fmt_date(state['session_date'])}", ""]
    by_question = {a["question_id"]: a for a in state.get("assessments", [])}
    plan = state.get("session_plan", [])
    if plan:
        lines += ["Plan: " + " · ".join(f"{p['skill']} ({p['kind']}, d{p['difficulty']}, {p['reason']})" for p in plan), ""]
    for q in state.get("questions", []):
        a = by_question.get(q["id"])
        title = f"Follow-up to {q['parent_id']}" if q.get("parent_id") else q["id"]
        lines += [f"## {title} · {q['skill']} · {q['kind']} · difficulty {q['difficulty']}", "", q["prompt"], ""]
        if q.get("grounding"):
            lines += [f"_Based on: <{q['grounding']}>_", ""]
        if q["kind"] == "coding" and q.get("starter_code"):
            lines += ["Starter code:", "", "```python", q["starter_code"].rstrip(), "```", ""]
        if a is None:
            lines += ["_Not answered (session ended)._", ""]
            continue
        if a["action"] == "skipped":
            lines += [f"**Skipped** (hints used: {a.get('hints_used', 0)})", ""]
        else:
            if q["kind"] == "coding":
                lines += ["**My answer:**", "", "```python", a["answer"].rstrip(), "```", ""]
            else:
                lines += ["**My answer:**", "", *[f"> {line}" for line in a["answer"].strip().splitlines()], ""]
            tests = f" · tests passed {pct(a['tests_passed'])}" if a.get("tests_passed") is not None else ""
            lines += [f"**Score:** {a['score']:.2f}{tests} · hints used: {a.get('hints_used', 0)}", *criteria_lines(a),
                      "", f"**Feedback:** {a['feedback']}", ""]
            if a.get("misconceptions"):
                lines += ["**Misconceptions:**", *[f"- {m}" for m in a["misconceptions"]], ""]
        lines += ["<details><summary>Reference answer</summary>", "", q.get("reference_answer", ""), "", "</details>", ""]
    if state.get("quit"):
        lines += ["_I ended the session early._", ""]
    return "\n".join(lines).rstrip() + "\n"


def compose_report(state: CoachState) -> dict:
    """Render the report and transcript (ARCHITECTURE.md §8) to REPORTS_DIR and SESSIONS_DIR."""
    conn = memory.connect()
    try:
        trend = memory.skill_trend(conn)
    finally:
        conn.close()
    report = render_report(state, trend)
    day = state["session_date"].isoformat()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    report_path = config.REPORTS_DIR / f"{day}.md"
    report_path.write_text(report)
    result = {"report_markdown": report, "report_path": str(report_path)}
    if state.get("questions"):
        transcript = render_transcript(state)
        config.SESSIONS_DIR.mkdir(exist_ok=True)
        transcript_path = config.SESSIONS_DIR / f"{day}.md"
        transcript_path.write_text(transcript)
        result.update(transcript_markdown=transcript, transcript_path=str(transcript_path))
    return result


def update_memory(state: CoachState) -> dict:
    """Write opportunities, questions, assessments, readiness, check-in updates, and the plan to
    career.db; add question embeddings to the vector store."""
    session_date = state["session_date"]
    conn = memory.connect()
    try:
        memory.record_opportunities(conn, state.get("opportunities", []) + state.get("rejected", []))
        memory.record_posting_texts(conn, session_date, state.get("posting_texts", {}))
        memory.record_session(conn, session_date, state.get("questions", []), state.get("assessments", []))
        if state.get("readiness_after"):
            memory.record_readiness(conn, session_date, state["readiness_after"], set(state.get("assessed_after", [])))
        memory.update_task_status(conn, session_date, state.get("plan_updates", []))
        plan = memory.record_plan(conn, session_date, state.get("prep_plan", []))
    finally:
        conn.close()
    embeddings = embeddings_or_none()
    if embeddings is not None and state.get("questions"):
        memory.add_questions(memory.load_vector_store(embeddings), state["questions"])
    return {"prep_plan": plan}
