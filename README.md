# Agentic AI Coworkers

CMPE 256 (SJSU, Fall 2026) assignment: two personalized agentic AI coworkers built with LangGraph / LangChain.

| Agent | Goal | Status |
|---|---|---|
| [DailyTechUpgradeAgent](agent1_daily_tech_upgrade/) | Recommend, summarize, and explain the top 3 technical articles each day for a 15–20 minute learning session | Scaffolded ([architecture](agent1_daily_tech_upgrade/ARCHITECTURE.md)); nodes not implemented |
| nextLevelPersonalCoachForCareerLevel | Personalized career coach: find opportunities, assess readiness, run interactive practice, track progress | Not started |

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
