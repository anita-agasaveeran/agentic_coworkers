"""Paths, tunable settings, and model factories for DailyTechUpgradeAgent."""

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
SOURCES_PATH = AGENT_DIR / "sources.yaml"
DATA_DIR = AGENT_DIR / "data"
HISTORY_DB = DATA_DIR / "history.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
VECTOR_STORE_PATH = DATA_DIR / "summary_vectors.json"
BRIEFS_DIR = AGENT_DIR / "briefs"

# --- Candidate search --------------------------------------------------------
LOOKBACK_DAYS = 3                 # initial window (72 h)
BROADENED_LOOKBACK_DAYS = 7       # window used after broaden_queries
MAX_RESULTS_PER_QUERY = 8
NUM_QUERIES = (5, 8)              # min/max queries from plan_queries
ARXIV_QUERIES = 4                 # arXiv allows one request per 3 s, so keep this small
SCORE_BATCH_SIZE = 30             # candidates per scoring call
MAX_TEXT_CHARS = 120_000          # cap on article text sent for summarizing (logged when hit)

# --- Selection (ARCHITECTURE.md §7) ------------------------------------------
TOP_K = 3
SCORE_THRESHOLD = 0.6
MAX_RETRIES = 2
NEAR_DUPLICATE_SIMILARITY = 0.9
SCORE_WEIGHTS = {
    "relevance": 0.35,
    "depth": 0.20,
    "novelty": 0.20,
    "authority": 0.15,
    "recency": 0.10,
}
TIER_AUTHORITY = {1: 1.0, 2: 0.7}
INTEREST_SPREAD_MARGIN = 0.15     # max score given up to cover another profile interest

# --- Reading-time budget -----------------------------------------------------
WORDS_PER_MINUTE = 230
BRIEF_MINUTES = 5
SESSION_MINUTES = (15, 20)

# --- Models ------------------------------------------------------------------
# Provider-swappable "provider:model" strings for LangChain's init_* helpers.
# Don't pass temperature/top_p: current Anthropic models reject sampling params.
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


def load_sources() -> dict:
    return yaml.safe_load(SOURCES_PATH.read_text())
