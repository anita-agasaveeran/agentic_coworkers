# Agentic AI Coworkers

CMPE 256 (SJSU, Fall 2026) assignment: two personalized agentic AI coworkers built with LangGraph / LangChain.

| Agent | Goal | Status |
|---|---|---|
| [DailyTechUpgradeAgent](agent1_daily_tech_upgrade/) | Recommend, summarize, and explain the top 3 technical articles each day for a 15–20 minute learning session | Implemented and evaluated ([architecture](agent1_daily_tech_upgrade/ARCHITECTURE.md), [status](agent1_daily_tech_upgrade/README.md#implementation-status), [sample briefs](agent1_daily_tech_upgrade/briefs/), [evaluation](agent1_daily_tech_upgrade/eval_reports/eval_2026-11-01.md)) |
| [nextLevelPersonalCoachForCareerLevel](agent2_career_coach/) | Personalized career coach: find opportunities, assess readiness, run interactive practice, track progress | Scaffolded ([architecture](agent2_career_coach/ARCHITECTURE.md), [status](agent2_career_coach/README.md#implementation-status)); nodes not implemented |

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (or plain `venv` + `pip`).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env   # then add your API keys
```

## Deliverables (per agent)

- Agent goals
- Architecture diagram (workflow, tools, data sources, memory)
- Working prototype
- Sample results
- Evaluation of effectiveness, accuracy, and user progress
