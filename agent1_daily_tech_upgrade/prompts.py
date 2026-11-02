"""Prompt templates for the LLM nodes. Fill with str.format(**kwargs)."""

PLAN_QUERIES = """You plan today's search for a daily technical reading brief.

Reader profile:
{profile}

Learning progress (last 14 days):
{progress}

Write {min_queries}-{max_queries} web search queries that find recent, in-depth technical articles
for this reader. Spread them across the reader's interests, favoring the ones listed as
under-covered, avoid topics the reader marked "less like this", and keep each query specific
(a technique, system, or problem, not a buzzword).

Also write {arxiv_queries} arXiv queries of 2-3 keywords each (e.g. "agentic retrieval",
"generative recommendation"). arXiv matches every keyword against paper abstracts, so use
terms that appear in research abstracts, not product or framework names."""

BROADEN_QUERIES = """Today's search found only {num_passing} articles that met the quality bar.

Previous queries:
{queries}

Reader profile:
{profile}

Write {min_queries}-{max_queries} broader or rephrased web search queries that are likely to
surface in-depth technical articles on the same interests, and {arxiv_queries} arXiv queries of
2-3 keywords each that would appear in research abstracts."""

SCORE_CANDIDATES = """Score each candidate article for this reader.

Reader profile:
{profile}

Topics the reader has read about recently:
{recent_topics}

From the reader's ratings: topics they liked: {liked_topics}; topics they want less of: {disliked_topics}.
Raise relevance for articles on liked topics and lower it for articles on topics they want less of.

Candidates ([id] title | source | snippet):
{candidates}

For each candidate, using its id, give integer scores 1-5:
- relevance: fit with the reader's interests and career goals
- depth: 1 = announcement or marketing, 5 = detailed technical content (methods, architecture, results)
- novelty: how new this is for the reader, given the recent topics
Also give 1-3 short topic labels, and the one profile interest the article serves most, copied
exactly from the profile's "interests" keys (or "none"). Score only from the information given;
do not guess at content that isn't in the snippet."""

JUDGE_FAITHFULNESS = """You check a summary against its source text for factual support.

Source text ({text_source}):
{text}

Summary:
{summary}

Key takeaways:
{takeaways}

List every factual claim in the summary or takeaways that the source text does not support
(numbers, names, results, causal statements). Interpretation and advice addressed to the reader
are fine. Then score 1-5: 5 = every claim supported, 4 = one minor unsupported detail,
3 = several unsupported details, 2 = a central claim unsupported, 1 = mostly unsupported."""

SUMMARIZE_ARTICLE = """Summarize this article for the reader's {minutes}-minute reading slot.

Reader profile:
{profile}

Article: {title} ({source}, {published})
Text ({text_source}):
{text}

Write:
- summary: 4-6 sentences on the technical content (what was done, how, and the result)
- key_takeaways: 3-5 concrete takeaways
- why_it_matters: how this connects to the reader's specific goals or skill gaps
- think_about: one question to reflect on after reading
- reading_scope: which parts to read to fit about {minutes} minutes (e.g. "Read the Method and
  Results sections"), or "whole article" if it fits

Use only the text above. If it is a short snippet rather than the full article, say less rather
than filling gaps."""
