# Agent 1: DailyTechUpgradeAgent

Finds, ranks, and explains the top 3 recent technical articles from authoritative sources, personalized to my interests and reading history, as a 15–20 minute daily brief. Design: [ARCHITECTURE.md](ARCHITECTURE.md).

## Run

From this folder, after the repo-level setup in the [root README](../README.md):

```bash
../.venv/bin/python agent.py                 # today's brief, then asks for ratings
../.venv/bin/python agent.py --no-feedback   # skip the rating step
../.venv/bin/python agent.py --date 2026-10-29
../.venv/bin/python evaluate.py --days 7
../.venv/bin/python -m pytest
```

Briefs are written to `briefs/YYYY-MM-DD.md`. Personal history and checkpoints live in `data/` (gitignored).

## Configure

- `profile.json`: interests (weighted), skill levels, career goals, things to avoid.
- `sources.yaml`: allow-listed sources, authority tiers, feed URLs, arXiv categories.
- `config.py`: lookback window, score weights, threshold, retries, reading-time budget.
- `.env` (repo root): `CHAT_MODEL`, `EMBEDDING_MODEL`, API keys.

## Implementation status

| Part | Status |
|---|---|
| Graph wiring, routing, feedback interrupt, CLI | done |
| State and structured-output schemas | done |
| SQLite schema | done |
| Prompts | drafted |
| Tools (`tools.py`) | done; feeds, arXiv, page loading tested live; `search_web` tested with a mocked Tavily client only |
| Nodes (`nodes.py`, except routing and feedback) | stubs |
| Memory read/write helpers | stubs |
| Evaluation (`evaluate.py`) | stubs |
| Source feed URLs (`sources.yaml`) | checked 2026-10-29 (10 feeds) |
