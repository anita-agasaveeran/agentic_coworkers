# DailyTechUpgradeAgent — Architecture

## 1. Goal

**Purpose:** Help me become a better technical learner every day by finding, ranking, and explaining the three most useful technical articles published recently by authoritative sources.

**Expected outcome (each day):** A *Daily Tech Brief* containing the top 3 articles, each with:
- title, source, publication date, URL (verified, never LLM-generated)
- a technical summary and 3–5 key takeaways
- "Why this matters for me" — link to my interests, skill gaps, and career goals
- estimated reading time, so that the whole session fits in **15–20 minutes**

Over time the agent learns from my feedback and reading history, avoids repeats, and steers toward topics I haven't covered yet.

---

## 2. High-level workflow

The agent is a **LangGraph `StateGraph`**: a deterministic pipeline with one retry loop and one human-in-the-loop step. LLM calls are used only where judgment is needed (ranking, summarizing, personalizing); fetching, filtering, and metadata are plain code so references stay accurate.

```mermaid
flowchart TD
    START([START / daily trigger]) --> P[load_profile]
    P --> Q[plan_queries]
    Q --> F[fetch_candidates]
    F --> D[dedupe_and_filter]
    D --> S[score_and_rank]
    S -->|"< 3 candidates pass threshold<br/>and retries < 2"| R[broaden_queries]
    R --> F
    S -->|"top 3 selected"| X[extract_full_text]
    X --> M[summarize_and_personalize]
    M --> B[compose_brief]
    B --> H{{collect_feedback<br/>human-in-the-loop}}
    H --> U[update_memory]
    U --> END([END → brief saved to briefs/YYYY-MM-DD.md])

    subgraph Tools
        T1[RSS / Atom feed reader]
        T2[Tavily web search<br/>domain-restricted]
        T3[arXiv API]
        T4[Web page loader<br/>requests + BeautifulSoup]
    end

    subgraph Memory
        M1[(profile.json<br/>interests, goals, level)]
        M2[(SQLite: history.db<br/>seen articles, feedback,<br/>topic progress)]
        M3[(Vector store<br/>embeddings of past summaries)]
        M4[(LangGraph checkpointer<br/>per-run state)]
    end

    F -.-> T1 & T2 & T3
    X -.-> T4
    P -.-> M1 & M2
    D -.-> M2 & M3
    S -.-> M2
    U -.-> M2 & M3
    START -.-> M4
```

---

## 3. Graph state

```python
class Article(TypedDict):
    url: str
    title: str
    source: str            # e.g. "Anthropic", "AWS ML Blog", "IEEE Spectrum"
    published: date
    snippet: str           # from feed/search result
    full_text: str | None
    word_count: int | None
    topics: list[str]
    scores: dict[str, float]   # relevance, authority, novelty, depth, recency
    total_score: float

class BriefState(TypedDict):
    run_date: date
    profile: dict                  # loaded from profile.json + derived progress
    queries: list[str]
    candidates: list[Article]      # raw fetched
    filtered: list[Article]        # after dedupe/filter
    selected: list[Article]        # top 3
    summaries: list[dict]          # per-article summary, takeaways, why_it_matters, read_minutes
    brief_markdown: str
    feedback: list[dict]           # ratings collected after the brief
    retries: int
```

---

## 4. Nodes

| Node | Type | What it does | Reads / writes |
|---|---|---|---|
| `load_profile` | code | Loads static interests and goals from `profile.json`; derives *learning progress* from SQLite (topics covered recently, topics rated "want more", under-covered interests). | M1, M2 → `profile` |
| `plan_queries` | LLM | Turns the profile into 5–8 web search queries spread across interests (weighting under-covered ones), plus 4 short 2–3 keyword arXiv queries (e.g. "agentic retrieval"). | `profile` → `queries`, `arxiv_queries` |
| `fetch_candidates` | tools | Pulls the last 72 h of posts from source feeds and arXiv; runs domain-restricted Tavily searches for each query. Captures title, URL, date, and source **from the tool response**. | → `candidates` |
| `dedupe_and_filter` | code | Normalizes URLs; drops items already in `history.db`; drops near-duplicates (embedding similarity > 0.9 to past summaries); drops non-allow-listed domains, items with no date, and items older than the window. | M2, M3 |
| `score_and_rank` | LLM + code | LLM scores each candidate 1–5 on relevance, depth, and novelty using structured output; code adds authority (source tier) and recency. Applies weighted sum plus a diversity rule (≤1 article per source, distinct topics). | → `selected` |
| `broaden_queries` | LLM | Runs only if fewer than 3 candidates pass the threshold. Widens the time window (72 h → 7 days) and/or rewrites queries. Max 2 retries; then falls back to the best available. | `retries += 1` |
| `extract_full_text` | tool | Fetches each selected page and extracts the main text (bs4, or Tavily extract for sites that block scripts). An article with no retrievable text is replaced by the next full-text candidate; it is used, flagged "snippet-based", only when no such candidate is left. | → `full_text`, `word_count` |
| `summarize_and_personalize` | LLM | For each article: a 4–6 sentence technical summary, 3–5 key takeaways, "why this matters for me" (tied to specific profile goals), one follow-up question to think about, and a suggested reading scope that fits the time budget. | → `summaries` |
| `compose_brief` | code (template) | Renders the Markdown brief: header, total time, 3 article cards, references section. Title, date, source, and URL are inserted from `Article` fields, not LLM output. | → `brief_markdown` |
| `collect_feedback` | human-in-the-loop | LangGraph `interrupt()`: asks for a 1–5 rating per article and optional "more/less like this" tags. Skippable (defaults to no feedback). | → `feedback` |
| `update_memory` | code | Writes the selected articles, topics, and feedback to SQLite; adds summary embeddings to the vector store; updates per-topic exposure counts. Saves the brief to `briefs/YYYY-MM-DD.md`. | M2, M3 |

---

## 5. Tools and data sources

**Tools** (each wrapped with `@tool` so they can also be called by an LLM agent if needed):

| Tool | Purpose |
|---|---|
| `fetch_feeds(sources, since)` | Reads RSS/Atom feeds with `feedparser`; returns `Article` records with tracking params stripped. |
| `search_web(query, sources, days)` | Tavily news search over the last `days` days, restricted to the sources' domains; drops search/listing pages. (General search with a start date returned nothing in testing.) |
| `search_arxiv(query, categories, since)` | arXiv API abstract search on up to 4 keywords (longer AND-queries timed out), newest first; respects the 3-second rate limit and retries once on HTTP 429. |
| `load_page(url)` | `requests` + BeautifulSoup main-content extraction (largest `<article>`, else `<main>`), falling back to Tavily extract when a site blocks scripts (e.g. openai.com); also reads the publication date from meta tags. Returns `None` if neither yields enough text. |
| `url_resolves(url)` | Link check. 401/403/429 count as resolving (the page exists but blocks scripts). |

**Source allow-list with authority tiers** (stored in `sources.yaml`; feeds checked 2026-10-29):

| Tier | Source | Access method |
|---|---|---|
| 1 | OpenAI | RSS (pages return 403 to scripts, so summaries use the feed snippet) |
| 1 | Google Research, Google AI blog, DeepMind | RSS (3 feeds) |
| 1 | Microsoft Research, AWS ML Blog, IBM Research, LangChain/LangGraph, IEEE Spectrum (AI) | RSS |
| 1 | Anthropic, ACM (CACM, Queue) | Tavily with `include_domains` (no feed / feeds return 403) |
| 1 | arXiv (cs.AI, cs.CL, cs.LG, cs.IR) | arXiv API |
| 2 | Hugging Face, NVIDIA Technical Blog | RSS |
| 2 | Meta AI | Tavily (no feed) |

---

## 6. Memory design

| Store | Kind | Contents | Used for |
|---|---|---|---|
| `profile.json` | long-term, user-edited | interests (weighted), current skill level per area, career goals (e.g. "ML engineer internship, summer 2027"), preferred depth | query planning, relevance scoring, "why it matters" |
| `history.db` (SQLite) | long-term, agent-written | `articles_seen(url, title, source, published, topics, score, shown_on)`<br>`feedback(url, rating, tags, date)`<br>`topic_progress(topic, exposures, last_seen, avg_rating)` | dedupe, learning-progress signal, evaluation metrics |
| Vector store (`InMemoryVectorStore`, persisted to `data/summary_vectors.json`) | long-term semantic | embeddings of past article summaries | novelty scoring, near-duplicate filtering |
| LangGraph checkpointer (`SqliteSaver` / `InMemorySaver`) | short-term, per run | full `BriefState` per `thread_id = run_date` | resuming after `interrupt()`, debugging, re-running a day |

**How personalization uses memory:**
- Topics with high `avg_rating` get a relevance boost; topics tagged "less like this" get a penalty.
- Interests with low `exposures` in the last 14 days get extra queries in `plan_queries` (spreading coverage).
- Articles too similar to anything already read get a lower novelty score.

---

## 7. Ranking and time budget

**Score** (each component normalized 0–1):

```
total = 0.35·relevance + 0.20·depth + 0.20·novelty + 0.15·authority + 0.10·recency
```

Selection = top 3 by `total`, subject to: at most one article per source, three distinct primary topics, and `total ≥ 0.6` (otherwise → `broaden_queries`). To spread the brief across profile interests, each pick goes to an article for a not-yet-covered interest when it scores within 0.15 (`INTEREST_SPREAD_MARGIN`) of the best remaining article.

**15–20 minute budget:** reading time = `word_count / 230 wpm`. The brief itself takes about 5 minutes; the remaining ~12 minutes are split across the 3 articles. For long articles or papers, `summarize_and_personalize` names the specific sections to read (e.g. "Read §3 Method and §5 Results") instead of the whole piece. The brief header shows the total estimated time.

---

## 8. Output format (`briefs/YYYY-MM-DD.md`)

```
# Daily Tech Brief — Oct 29, 2026   (≈18 min)
Focus today: agent memory · RAG evaluation · cloud inference cost

## 1. <Title>
Source: <Source> · Published: <date> · <URL> · ~5 min
Summary: ...
Key takeaways: • ... • ... • ...
Why this matters for you: ...
What to read: <sections that fit the time slot>
Think about: <one question>

## 2. ...
## 3. ...

### References
1. <Title>. <Source>, <date>. <URL>
...
```

---

## 9. Reliability and guardrails

- **No invented references:** metadata comes only from tool results; `compose_brief` checks that every summary belongs to a `selected` article. The scorer refers to candidates by list number, not URL, so the LLM never writes a URL.
- **Structured outputs:** LLM nodes use `with_structured_output(..., method="json_schema")` (native structured outputs), since Claude Opus 5.5 rejects forced tool use. Score ranges and list lengths are enforced in code.
- **Link check:** an HTTP HEAD/GET on each selected URL before summarizing; broken links (404/410, 5xx, no connection) are swapped for the next-ranked candidate. Pages that block scripts (403) are kept and summarized from the feed snippet.
- **Grounded summaries:** the summarization prompt includes only the extracted text; if extraction failed, the card is labeled "summary based on abstract/snippet".
- **Fallbacks:** feed failure → Tavily only; Tavily failure or no key → feeds + arXiv only; no embeddings key → near-duplicate check skipped; < 3 results after retries → publish fewer articles with a note instead of padding.
- **Tracing:** LangSmith tracing enabled for debugging and evaluation.

---

## 10. Evaluation (`evaluate.py`)

`evaluate.py` reads every saved run (the final graph state per run date from the LangGraph checkpointer) plus `history.db`, and writes a Markdown report to `eval_reports/eval_YYYY-MM-DD.md`. `--no-llm` skips the judge; `--no-network` skips live page checks.

| Area | Metrics | How it is measured |
|---|---|---|
| Pipeline health | candidates per run, full briefs (3 articles), runs needing a retry, full-text rate, mean selected score | saved run state |
| Reference accuracy | briefs whose links and References exactly match the selected articles; links resolving; titles and dates matching the live page (±2 days) | offline check on the brief text, then a fetch of each article page; pages that block scripts are counted as unverifiable |
| Summary faithfulness | mean 1–5 support score; share scored 4–5; list of unsupported claims | LLM-as-judge (`JUDGE_FAITHFULNESS`) compares each summary and its takeaways to the text the summarizer saw. The judge is the same model family as the summarizer, so the scores can run high; the claim list is the more useful output |
| Relevance | articles rated, mean rating, precision@3 (share rated ≥ 4) | `feedback` table (your ratings) |
| Novelty and diversity | repeated articles across briefs, distinct sources and topics, per-brief source diversity, profile interests covered | saved runs |
| Time budget | mean brief minutes, share within 15–20 min | `read_minutes` per summary + 5 min for the brief |
| Learning progress | articles per interest, mean rating by day, liked/disliked topics, interests the agent will steer toward next | saved runs, `feedback`, `topic_progress` |

---

## 11. Tech stack and file layout

**Stack:** Python 3.12, LangGraph, LangChain (`init_chat_model` / `init_embeddings`, so the provider is set by a `provider:model` string in `.env`; default chat model `anthropic:claude-opus-5-5`, default embeddings `openai:text-embedding-3-small`), Tavily, feedparser, requests + BeautifulSoup, SQLite, `InMemoryVectorStore` persisted to JSON, LangSmith. A daily run can be scheduled with cron.

```
agent1_daily_tech_upgrade/
├── ARCHITECTURE.md        # this file
├── profile.json           # my interests, level, goals
├── sources.yaml           # allow-listed sources, tiers, feed URLs
├── config.py              # paths, weights/thresholds, model factories
├── state.py               # BriefState, Article, structured-output schemas
├── nodes.py               # one function per graph node + routing
├── agent.py               # StateGraph wiring + CLI (feedback interrupt loop)
├── tools.py               # fetch_feeds, search_web, search_arxiv, load_page, url_resolves
├── memory.py              # SQLite schema/helpers + vector store helpers
├── prompts.py             # planning, scoring, summarizing prompts
├── evaluate.py            # offline metrics over saved runs
├── demo.ipynb             # working prototype walkthrough
├── tests/                 # pytest (fakes.py + conftest.py: fake LLM, embeddings, network)
├── briefs/                # generated daily briefs (sample results)
├── eval_reports/          # evaluate.py output
└── data/                  # history.db, checkpoints.db, summary_vectors.json (gitignored)
```
