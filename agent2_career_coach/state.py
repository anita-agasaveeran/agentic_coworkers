"""Graph state and structured-output schemas for nextLevelPersonalCoachForCareerLevel."""

from datetime import date
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

OpportunityType = Literal["internship", "job", "research", "entrepreneurial"]
QuestionKind = Literal["concept", "coding", "scenario"]


class SkillReq(TypedDict):
    skill: str               # id from skills.yaml, or "other:<raw name>"
    level: int               # 1 = familiarity, 2 = working knowledge, 3 = strong / production
    required: bool           # required vs preferred
    evidence: str            # quoted line from the posting


class Opportunity(TypedDict, total=False):
    url: str
    title: str               # from the search result / page, never the LLM
    type: OpportunityType
    source: str              # display name from opportunity_sources.yaml
    snippet: str
    org: str | None
    location: str | None
    deadline: date | None    # as stated on the page; None if not stated
    found_on: date
    requirements: list[SkillReq]
    fit_score: float         # 0-1
    status: str              # open | closed | dismissed (closed / not a posting / ineligible / unreadable)


class Question(TypedDict, total=False):
    id: str                  # "<session date>-q<N>", follow-ups "-q<N>f"
    skill: str
    kind: QuestionKind
    difficulty: int          # 1-3
    prompt: str
    grounding: str | None    # posting requirement or article the question is based on (URL)
    reference_answer: str    # hidden from me; used by the grader
    rubric: list[str]
    hints: list[str]         # shown one at a time on request
    starter_code: str | None     # coding only
    tests: str | None            # coding only (pytest), validated against the reference solution
    workspace_file: str | None   # coding only: file I edit
    parent_id: str | None        # set on follow-ups


class Assessment(TypedDict, total=False):
    question_id: str
    action: Literal["answered", "skipped"]
    answer: str
    hints_used: int
    criteria_scores: dict[str, int]   # rubric criterion -> 1-5
    tests_passed: float | None        # coding only, 0-1
    score: float                      # 0-1, computed in code (ARCHITECTURE.md §7)
    feedback: str
    misconceptions: list[str]


class Resource(TypedDict):
    url: str
    title: str
    source: str
    skill: str
    why: str


class PlanTask(TypedDict, total=False):
    id: int                  # career.db plan_tasks id (set when stored)
    task: str
    skill: str
    resource_url: str | None
    hours: float
    due: date
    done_when: str
    status: Literal["open", "partial", "done", "dropped"]


class CoachState(TypedDict, total=False):
    session_date: date
    interactive: bool                 # False skips the check-in and asks no questions
    refresh: bool                     # force opportunity discovery
    extra_postings: list[str]         # URLs from --add-posting
    session_size: int                 # questions planned (0 = report only)
    profile: dict                     # profile.json + resume + derived progress
    discovery_due: bool               # opportunities stale, missing, or --refresh
    open_tasks: list[PlanTask]        # from the last plan, for the check-in
    plan_updates: list[dict]          # {"id", "status"} from the check-in
    queries: list[dict]               # {"type", "query"}
    candidates: list[Opportunity]     # newly discovered, not yet analyzed
    opportunities: list[Opportunity]  # stored + newly analyzed, ranked by fit
    rejected: list[Opportunity]       # analyzed and dropped (stored so they aren't analyzed again)
    skill_demand: dict[str, float]    # skill -> 0-1
    skill_targets: dict[str, int]     # skill -> highest level asked (1-3)
    readiness: dict[str, float]       # skill -> 0-1 before this session
    assessed_skills: list[str]        # skills with assessments (others are self-rated)
    previous: dict                    # {"date", "readiness"} from the last session, for deltas
    weak_skills: list[str]            # latest session's mean score below PROBE_BELOW (retest candidates)
    session_plan: list[dict]          # [{"skill", "kind", "difficulty"}]
    questions: list[Question]
    pending_answer: dict              # resume value from ask_and_collect_answer
    assessments: list[Assessment]
    quit: bool                        # I ended the question loop early
    readiness_after: dict[str, float]
    assessed_after: list[str]
    strengths: list[dict]             # {"skill", "summary", "evidence": [...]}
    gaps: list[dict]
    requirement_refs: dict[str, str]  # "R<n>" -> posting requirement cited by strengths and gaps
    resources: list[Resource]
    prep_plan: list[PlanTask]
    report_markdown: str
    report_path: str
    transcript_markdown: str
    transcript_path: str


# --- Structured LLM outputs --------------------------------------------------
# As in Agent 1: ranges and list lengths are stated in descriptions and enforced in
# code, not as JSON Schema constraints, so the schemas work with native structured outputs.

class OpportunityQuery(BaseModel):
    type: OpportunityType
    query: str = Field(description="Web search query for open postings of this type")


class OpportunityQueries(BaseModel):
    queries: list[OpportunityQuery]


class SkillRequirement(BaseModel):
    skill: str = Field(description="Skill id from the taxonomy, or 'other:<name>' if none fits")
    level: int = Field(description="1 = familiarity, 2 = working knowledge, 3 = strong / production experience")
    required: bool = Field(description="True if listed as required, False if preferred / nice to have")
    evidence: str = Field(description="The exact line from the posting that states this requirement")


class PostingAnalysis(BaseModel):
    is_posting: bool = Field(description="False if the page is not a single opportunity (listing, article, error page)")
    title: str | None = Field(description="Role or program title copied exactly from the page")
    type: OpportunityType = Field(description="internship, job, research (positions and graduate programs), or entrepreneurial")
    org: str | None = Field(description="Hiring organization, program, or lab as written on the page")
    location: str | None
    deadline: str | None = Field(description="Application deadline as ISO date if the page states one, else null")
    closed: bool = Field(description="True if the page says applications are closed")
    eligible: bool = Field(description="False if the candidate clearly fails a stated hard requirement (degree, timing, authorization)")
    responsibilities: list[str] = Field(description="3-6 main responsibilities or research activities")
    requirements: list[SkillRequirement]


class GeneratedQuestion(BaseModel):
    prompt: str = Field(description="The question as asked to the candidate")
    grounding_id: int | None = Field(description="Number of the posting requirement or article it is based on, or null")
    reference_answer: str = Field(description="A strong answer, hidden from the candidate")
    rubric: list[str] = Field(description="3-4 criteria the answer is graded on")
    hints: list[str] = Field(description="Progressive hints, least revealing first")
    starter_code: str | None = Field(description="Coding only: function signature and docstring to complete")
    tests: str | None = Field(description="Coding only: pytest tests that import from 'solution'")
    reference_solution: str | None = Field(description="Coding only: a correct solution that passes the tests")


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(description="1-5")


class Grade(BaseModel):
    criteria: list[CriterionScore]
    feedback: str = Field(description="2-4 sentences: what was right, what was missing, how to improve")
    misconceptions: list[str] = Field(description="Specific errors or misunderstandings in the answer")


class Finding(BaseModel):
    skill: str
    summary: str
    evidence: list[str] = Field(description="Question ids or posting requirements that support this")


class StrengthsGaps(BaseModel):
    strengths: list[Finding]
    gaps: list[Finding]


class ResourcePick(BaseModel):
    id: int = Field(description="The resource's number from the list")
    skill: str
    why: str = Field(description="One sentence on what it fixes for this candidate")


class TaskDraft(BaseModel):
    task: str
    skill: str
    resource_id: int | None = Field(description="Number of a picked resource, or null")
    hours: float
    due_in_days: int
    done_when: str = Field(description="A concrete check that shows the task is done")


class PrepPlanOutput(BaseModel):
    resources: list[ResourcePick]
    tasks: list[TaskDraft]


class QuestionJudgment(BaseModel):
    relevance: int = Field(description="1-5: tests the stated skill and gap")
    difficulty_fit: int = Field(description="1-5: matches the planned difficulty")
    clarity: int = Field(description="1-5: unambiguous and answerable in a few minutes")
