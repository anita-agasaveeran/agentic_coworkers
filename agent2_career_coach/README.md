# Agent 2: nextLevelPersonalCoachForCareerLevel

Personal career coach: finds internships, jobs, research, and entrepreneurial opportunities, extracts the skills they ask for, quizzes me interactively on my biggest gaps, and tracks my readiness over time in a Career Readiness Report. Design: [ARCHITECTURE.md](ARCHITECTURE.md). Deliverables walkthrough: [demo.ipynb](demo.ipynb).

## Run

From this folder, after the repo-level setup in the [root README](../README.md):

```bash
../.venv/bin/python agent.py                         # coaching session; discovers opportunities if stale
../.venv/bin/python agent.py --refresh               # force opportunity discovery
../.venv/bin/python agent.py --questions 3           # shorter session
../.venv/bin/python agent.py --add-posting <url>     # analyze a posting found elsewhere (repeatable)
../.venv/bin/python agent.py --report-only           # report from stored data, no questions
../.venv/bin/python evaluate.py                      # writes eval_reports/eval_YYYY-MM-DD.md
../.venv/bin/python evaluate.py --no-llm --no-network   # free, offline metrics only
../.venv/bin/python -m pytest
```

During a session, each question gets a file in `workspace/`: write text answers below the marker line in `<question id>.md`, and code in `<question id>.py`, then save and press Enter in the terminal. Type `hint`, `skip`, or `quit` at the `>` prompt. If the terminal is closed mid-session, running `agent.py` again the same day resumes it. A finished session isn't run again for the same date, since that would overwrite its questions and report; use `--date YYYY-MM-DD` for another day.

Reports go to `reports/YYYY-MM-DD.md` and session transcripts to `sessions/YYYY-MM-DD.md`. History and checkpoints live in `data/` (gitignored).

## Configure

- `profile.json`: targets (role, type, timeline, priority), constraints, self-rated skills (ids from `skills.yaml`), projects.
- `resume.md` (optional, gitignored): plain-text resume used as extra context.
- `opportunity_sources.yaml`: allow-listed job, research, and entrepreneurial domains; learning-resource domains; arXiv categories.
- `skills.yaml`: canonical skill ids and aliases.
- `config.py`: discovery window, session size, follow-up thresholds, scoring constants, plan length.
- `.env` (repo root): the same `CHAT_MODEL`, `EMBEDDING_MODEL`, and API keys as Agent 1.

Page loading, link checks, and arXiv search reuse Agent 1's `tools.py`. Agent 1's `data/history.db` is read (read-only) for reading history when it exists.

## Implementation status

| Part | Status |
|---|---|
| Graph wiring, routing, check-in and question interrupts, CLI, session resume | done |
| State and structured-output schemas | done |
| SQLite schema | done |
| Prompts | drafted |
| Agent 1 tool reuse (`load_page`, `url_resolves`, `search_arxiv`) | done |
| New tools (`search_opportunities`, `search_resources`, `load_posting`, `posting_is_open`, `run_tests`) | done; tested offline and checked live 2026-11-01 |
| Nodes and scoring (§7) | done; tested with a fake LLM; sample session 2026-11-01 ([report](reports/2026-11-01.md), [transcript](sessions/2026-11-01.md)) |
| Memory read/write helpers | done |
| Evaluation (`evaluate.py`) | done; tested offline, run on the sample session ([report](eval_reports/eval_2026-11-01.md)) |
| Calibration set (`eval/calibration.jsonl`) | 15 labeled answers; partial and wrong labels scored by rubric criterion after the first grader run (see ARCHITECTURE §10) |
| Opportunity source domains | checked 2026-11-01 |
| Demo notebook (`demo.ipynb`) | covers the five deliverables; executed on the 2026-11-01 sample session |

## API keys

| Key | Needed for | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` (or another `CHAT_MODEL` provider) | extraction, questions, grading, coaching | the agent can't run |
| `TAVILY_API_KEY` | opportunity and learning-resource search | only stored opportunities and `--add-posting` URLs are used |
| `OPENAI_API_KEY` | embeddings for the repeat-question check | repeats are checked by exact text only |
