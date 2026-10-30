"""Offline evaluation over saved runs (ARCHITECTURE.md §10).

    python evaluate.py               # all runs
    python evaluate.py --days 7      # last 7 runs
"""

import argparse


def reference_accuracy(runs: list[dict]) -> float:
    """Share of brief URLs that resolve and whose page title/date match the stored metadata."""
    raise NotImplementedError


def summary_faithfulness(runs: list[dict]) -> float:
    """LLM-as-judge: mean 1-5 score that each takeaway is supported by the article text."""
    raise NotImplementedError


def relevance(conn) -> dict:
    """Mean user rating and precision@3 (share of articles rated >= 4)."""
    raise NotImplementedError


def novelty_and_diversity(conn, runs: list[dict]) -> dict:
    """Share of articles below the near-duplicate threshold; distinct sources/topics per week."""
    raise NotImplementedError


def time_budget_fit(runs: list[dict]) -> float:
    """Share of briefs whose estimated total time is within config.SESSION_MINUTES."""
    raise NotImplementedError


def learning_progress(conn) -> dict:
    """Topic coverage vs. profile interests over time, and the trend in ratings."""
    raise NotImplementedError


def load_runs(days: int | None) -> list[dict]:
    """Final BriefState of each saved run, read from the LangGraph checkpointer (thread_id = run date)."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Evaluate DailyTechUpgradeAgent runs.")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    # TODO: load runs + history.db, compute each metric, print a table and write data/eval_report.md
    raise NotImplementedError(f"evaluation not implemented yet (days={args.days})")


if __name__ == "__main__":
    main()
