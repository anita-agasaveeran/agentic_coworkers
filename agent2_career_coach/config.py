"""Paths, tunable settings, and model factories for nextLevelPersonalCoachForCareerLevel."""

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(AGENT_DIR.parent / ".env")

# --- Paths -------------------------------------------------------------------
PROFILE_PATH = AGENT_DIR / "profile.json"
RESUME_PATH = AGENT_DIR / "resume.md"                 # optional, plain text (gitignored)
OPPORTUNITY_SOURCES_PATH = AGENT_DIR / "opportunity_sources.yaml"
SKILLS_PATH = AGENT_DIR / "skills.yaml"
DATA_DIR = AGENT_DIR / "data"
CAREER_DB = DATA_DIR / "career.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
VECTOR_STORE_PATH = DATA_DIR / "question_vectors.json"
REPORTS_DIR = AGENT_DIR / "reports"
SESSIONS_DIR = AGENT_DIR / "sessions"
WORKSPACE_DIR = AGENT_DIR / "workspace"               # coding exercise files
AGENT1_DIR = AGENT_DIR.parent / "agent1_daily_tech_upgrade"
AGENT1_HISTORY_DB = AGENT1_DIR / "data" / "history.db"

# --- Opportunity discovery ---------------------------------------------------
OPPORTUNITY_MAX_AGE_DAYS = 7      # rediscover when the newest stored opportunity is older
SEARCH_DAYS = 30                  # posting recency window for web search
NUM_QUERIES = (4, 8)              # min/max queries from plan_opportunity_queries
MAX_RESULTS_PER_QUERY = 8
MAX_OPPORTUNITIES = 10            # top opportunities by fit used for skill demand and the report
MAX_POSTING_CHARS = 30_000        # cap on posting text sent for skill extraction

# --- Coaching session (ARCHITECTURE.md §4) -----------------------------------
SESSION_QUESTIONS = 5
REQUIRED_KINDS = ("coding", "scenario")   # at least one of each per session
MAX_FOLLOW_UPS = 1                # per question
PROBE_BELOW = 0.6                 # follow up on the misconception below this score
CHALLENGE_AT = 0.85               # follow up with a harder variant at or above this score
HINTS_PER_QUESTION = 2
HINT_PENALTY = 0.1
REPEAT_SIMILARITY = 0.9           # reject generated questions this similar to a past one
CODE_TIMEOUT_SECONDS = 10

# --- Scoring (ARCHITECTURE.md §7) --------------------------------------------
CODING_WEIGHTS = {"tests": 0.6, "review": 0.4}
DIFFICULTY_FACTOR = {1: 0.6, 2: 0.8, 3: 1.0}
TARGET_READINESS = {1: 0.5, 2: 0.7, 3: 0.85}
SELF_RATED_READINESS = {"beginner": 0.3, "intermediate": 0.5, "advanced": 0.7}
REQUIRED_WEIGHT = {True: 1.0, False: 0.5}
READINESS_HALF_LIFE_DAYS = 21
READING_BONUS = 0.05              # max readiness added from Agent 1 reading
DIFFICULTY_BANDS = ((0.4, 1), (0.7, 2), (1.01, 3))   # readiness below bound -> question difficulty

# --- Preparation plan --------------------------------------------------------
PLAN_DAYS = 14
PLAN_GAPS = 4                     # gaps covered by resources and plan tasks
RESOURCES_PER_GAP = 2

# --- Models ------------------------------------------------------------------
# Same provider-swappable settings as Agent 1. Don't pass temperature/top_p:
# current Anthropic models reject sampling params.
CHAT_MODEL = os.getenv("CHAT_MODEL", "anthropic:claude-opus-5-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small")


@lru_cache
def get_llm():
    from langchain.chat_models import init_chat_model

    return init_chat_model(CHAT_MODEL)


@lru_cache
def get_embeddings():
    from langchain.embeddings import init_embeddings

    return init_embeddings(EMBEDDING_MODEL)


def load_profile_file() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def load_resume() -> str | None:
    return RESUME_PATH.read_text() if RESUME_PATH.exists() else None


def load_opportunity_sources() -> dict:
    return yaml.safe_load(OPPORTUNITY_SOURCES_PATH.read_text())


def load_skills() -> dict:
    return yaml.safe_load(SKILLS_PATH.read_text())
