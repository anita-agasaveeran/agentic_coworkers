# Agent 1: DailyTechUpgradeAgent

Finds, ranks, and explains the top 3 recent technical articles from authoritative sources, personalized to my interests and reading history, as a 15–20 minute daily brief. Design: [ARCHITECTURE.md](ARCHITECTURE.md).

**Deliverables walkthrough:** [demo.ipynb](demo.ipynb) covers the agent goals, architecture diagram, a live run of the prototype, sample results and the evaluation, with saved outputs.

## Run

From this folder, after the repo-level setup in the [root README](../README.md):

```bash
../.venv/bin/python agent.py                 # today's brief, then asks for ratings
../.venv/bin/python agent.py --no-feedback   # skip the rating step
../.venv/bin/python agent.py --date 2026-10-29
../.venv/bin/python agent.py --rate 2026-10-29  # rate a brief generated earlier
../.venv/bin/python evaluate.py              # report over all saved runs -> eval_reports/
../.venv/bin/python evaluate.py --days 7 --no-llm --no-network   # last 7 runs, offline checks only
../.venv/bin/python -m pytest
```

Briefs are written to `briefs/YYYY-MM-DD.md`. Personal history and checkpoints live in `data/` (gitignored).

## Sample results

Four real daily runs (2026-10-30 to 2026-11-02). The first three were rated afterwards; the fourth, generated in [demo.ipynb](demo.ipynb), is the first to use those ratings:

- Briefs: [2026-10-30](briefs/2026-10-30.md), [2026-10-31](briefs/2026-10-31.md), [2026-11-01](briefs/2026-11-01.md), [2026-11-02](briefs/2026-11-02.md)
- Evaluation: [eval_reports/eval_2026-11-01.md](eval_reports/eval_2026-11-01.md): 12/12 links resolve, every checkable title and date matches the page, LLM-judge faithfulness 4.33/5, precision@3 44%, all briefs within 15–20 minutes. The judge's scores vary a little between runs. Findings and limitations are in section 5.3 of the notebook.

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
| Tools (`tools.py`) | done; all tested live |
| Memory (`memory.py`) | done |
| Nodes (`nodes.py`) | done; full graph tested with a fake LLM and run live end to end (≈2 min per run) |
| Prompts | revised after live runs |
| Evaluation (`evaluate.py`) | done; tested offline and on live runs |
| Demo notebook (`demo.ipynb`) | done; executed with a live run, outputs saved |
| Source feed URLs (`sources.yaml`) | 20 feeds: AI/ML checked 2026-10-29, physics/astrophysics checked 2026-11-01 |

## API keys

| Key | Needed for | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` (or another `CHAT_MODEL` provider) | query planning, scoring, summaries | the agent can't run |
| `OPENAI_API_KEY` | embeddings for near-duplicate checks | near-duplicate check is skipped |
| `TAVILY_API_KEY` | Anthropic, Meta AI, ACM (no usable feeds) and query-targeted search | only feeds and arXiv are used |
