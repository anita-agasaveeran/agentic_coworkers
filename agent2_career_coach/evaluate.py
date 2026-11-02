"""Offline evaluation over saved coaching sessions (ARCHITECTURE.md §10).

    python evaluate.py                  # all saved sessions
    python evaluate.py --days 7         # sessions in the last 7 days
    python evaluate.py --no-llm         # skip the grader re-runs and question judge
    python evaluate.py --no-network     # skip live posting and link checks

Writes a Markdown report to eval_reports/eval_YYYY-MM-DD.md and prints it.
"""

import argparse
import sqlite3
from pathlib import Path

import config

REPORTS_DIR = config.AGENT_DIR / "eval_reports"
CALIBRATION_PATH = config.AGENT_DIR / "eval" / "calibration.jsonl"
REGRADE_RUNS = 3


def load_sessions(days: int | None = None, checkpoint_db: Path | None = None) -> list[dict]:
    """Final CoachState of each saved session (thread_id = session date), oldest first."""
    raise NotImplementedError


def opportunity_accuracy(conn: sqlite3.Connection, network: bool = True) -> dict:
    """Links resolving, postings still open, deadlines in the future, share matching target type."""
    raise NotImplementedError


def requirement_extraction(conn: sqlite3.Connection, sessions: list[dict]) -> dict:
    """Share of evidence quotes found in the posting text; share of skills mapped to the taxonomy;
    the most frequent unmapped "other:" skills."""
    raise NotImplementedError


def grader_accuracy(calibration_path: Path = CALIBRATION_PATH, runs: int = REGRADE_RUNS) -> dict:
    """Spearman correlation and mean absolute error between grader scores and my labels on the
    calibration set, and the mean score spread when re-grading the same answer `runs` times."""
    raise NotImplementedError


def question_quality(sessions: list[dict], use_llm: bool = True) -> dict:
    """Mean relevance / difficulty fit / clarity (LLM judge, QuestionJudgment); repeated questions;
    coding questions whose tests pass the reference solution."""
    raise NotImplementedError


def coaching_effect(conn: sqlite3.Connection) -> dict:
    """Share of probe follow-ups scoring higher than their parent; misconceptions later answered well."""
    raise NotImplementedError


def user_progress(conn: sqlite3.Connection) -> dict:
    """Readiness per skill and per target over sessions; plan tasks done on time; gaps that became strengths."""
    raise NotImplementedError


def report_grounding(sessions: list[dict]) -> dict:
    """Share of findings whose evidence ids exist; share of report links found in career.db."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Evaluate nextLevelPersonalCoachForCareerLevel sessions.")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()
    # TODO: load sessions + career.db, compute each metric, render and write REPORTS_DIR/eval_<date>.md
    raise NotImplementedError(f"evaluation not implemented yet (days={args.days})")


if __name__ == "__main__":
    main()
