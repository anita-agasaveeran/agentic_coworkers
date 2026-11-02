"""Offline evaluation over saved coaching sessions (ARCHITECTURE.md §10).

    python evaluate.py                  # all saved sessions
    python evaluate.py --days 7         # the last 7 sessions
    python evaluate.py --no-llm         # skip the grader re-runs and question judge
    python evaluate.py --no-network     # skip live posting and link checks

Writes a Markdown report to eval_reports/eval_YYYY-MM-DD.md and prints it.
"""

import argparse
import json
import re
import sqlite3
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import config
import memory
import nodes
import prompts
import tools
from state import QuestionJudgment

REPORTS_DIR = config.AGENT_DIR / "eval_reports"
CALIBRATION_PATH = config.AGENT_DIR / "eval" / "calibration.jsonl"
REGRADE_RUNS = 3
MAX_LIVE_CHECKS = 25              # open postings checked live per evaluation
GOOD_SCORE = 0.7                  # an answer this good counts as "answered well"
DISAGREE_BY = 0.25                # calibration items listed when the grader is this far from my label
REPEAT_RATIO = 0.9                # questions this similar (character match) count as repeats


# --- Loading -----------------------------------------------------------------

def load_sessions(days: int | None = None, checkpoint_db: Path | None = None) -> list[dict]:
    """Final CoachState of each finished session (thread_id = session date), oldest first."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    import agent

    db = checkpoint_db or config.CHECKPOINT_DB
    if not db.exists():
        return []
    with SqliteSaver.from_conn_string(str(db)) as checkpointer:
        threads = sorted({c.config["configurable"]["thread_id"] for c in checkpointer.list(None)})
        app = agent.build_graph(checkpointer)
        sessions = [app.get_state({"configurable": {"thread_id": t}}).values for t in threads]
    sessions = [s for s in sessions if s.get("report_markdown")]
    return sessions[-days:] if days else sessions


def _ratio(n: int, d: int) -> str | None:
    return f"{n}/{d} ({n / d:.0%})" if d else None


def _mean(values, digits: int = 3):
    values = [v for v in values if v is not None]
    return round(mean(values), digits) if values else None


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\"“”'‘’`]", "", text or "").lower().split())


# --- Pipeline health ---------------------------------------------------------

def session_health(sessions: list[dict]) -> dict:
    questions = [q for s in sessions for q in s.get("questions", [])]
    assessments = [a for s in sessions for a in s.get("assessments", [])]
    answered = [a for a in assessments if a["action"] == "answered"]
    follow_ups = [q for q in questions if q.get("parent_id")]
    return {
        "sessions": len(sessions),
        "sessions_with_discovery": sum(bool(s.get("candidates")) for s in sessions),
        "questions_asked": f"{len(questions) - len(follow_ups)} + {len(follow_ups)} follow-ups",
        "answered": _ratio(len(answered), len(assessments)),
        "sessions_ended_early": sum(bool(s.get("quit")) for s in sessions),
        "hints_per_answer": _mean([a.get("hints_used", 0) for a in answered], 2),
        "mean_score": _mean([a["score"] for a in answered]),
        "coding_questions": sum(q["kind"] == "coding" for q in questions),
        "planned_coding_slots_asked_as_concept": sum(
            p["kind"] == "coding" and q["kind"] != "coding"
            for s in sessions for p, q in zip(s.get("session_plan", []), nodes.top_level_questions(s.get("questions", [])))),
    }


# --- Opportunity accuracy ----------------------------------------------------

def opportunity_accuracy(conn: sqlite3.Connection, today: date, target_types: set[str], network: bool = True) -> dict:
    """Filter counts, open postings' deadlines and target-type match; live: links resolving and
    postings still open (at most MAX_LIVE_CHECKS, best fit first)."""
    rows = conn.execute("SELECT * FROM opportunities ORDER BY fit_score DESC").fetchall()
    open_rows = [r for r in rows if r["status"] == "open"]
    with_deadline = [r for r in open_rows if r["deadline"]]
    statuses = Counter(r["status"] for r in rows)
    result = {
        "analyzed": len(rows),
        "kept_open": statuses["open"],
        "dropped_closed": statuses["closed"],
        "dropped_not_posting_or_ineligible": statuses["dismissed"],
        "open_by_type": dict(Counter(r["type"] for r in open_rows)),
        "open_matching_a_target_type": _ratio(sum(r["type"] in target_types for r in open_rows), len(open_rows)),
        "deadlines_in_future": _ratio(sum(r["deadline"] >= today.isoformat() for r in with_deadline), len(with_deadline)),
        "open_without_stated_deadline": len(open_rows) - len(with_deadline),
    }
    if not network or not open_rows:
        return result
    checked = open_rows[:MAX_LIVE_CHECKS]
    resolving, still_open, readable = 0, 0, 0
    for r in checked:
        if not tools.url_resolves(r["url"]):     # removed; Tavily's cached copy would still read as open
            readable += 1
            continue
        resolving += 1
        posting = tools.load_posting(r["url"])
        if posting is not None:
            readable += 1
            still_open += tools.posting_is_open(posting, memory._date(r["deadline"]), today)
    return result | {
        "links_resolving": _ratio(resolving, len(checked)),
        "still_open": _ratio(still_open, readable),
        "unreadable_now": len(checked) - readable,
    }


# --- Requirement extraction --------------------------------------------------

def evidence_found(evidence: str, text: str) -> bool:
    """The quoted line appears in the page text (case, quotes, and whitespace ignored; "..." splits it)."""
    page = _norm(text)
    parts = [p.strip(" .,;:") for p in re.split(r"\.\.\.|…", _norm(evidence))]
    parts = [p for p in parts if len(p) > 3]
    return bool(parts) and all(p in page for p in parts)


def requirement_extraction(conn: sqlite3.Connection) -> dict:
    """Share of evidence quotes found in the stored posting text; share of skills mapped to the
    taxonomy; the most frequent unmapped "other:" skills."""
    texts = memory.posting_texts(conn)
    rows = conn.execute("SELECT r.url, r.skill, r.evidence FROM requirements r JOIN opportunities o USING (url) "
                        "WHERE o.status = 'open'").fetchall()
    checkable = [r for r in rows if r["url"] in texts]
    missing = [r for r in checkable if not evidence_found(r["evidence"], texts[r["url"]])]
    unmapped = Counter(r["skill"].removeprefix("other:") for r in rows if r["skill"].startswith("other:"))
    per_posting = Counter(r["url"] for r in rows)
    return {
        "requirements": len(rows),
        "per_posting": _mean(per_posting.values(), 1),
        "evidence_found_in_posting": _ratio(len(checkable) - len(missing), len(checkable)),
        "mapped_to_taxonomy": _ratio(len(rows) - sum(unmapped.values()), len(rows)),
        "top_unmapped_skills": [f"{s} ({n})" for s, n in unmapped.most_common(10)],
        "evidence_not_found": [f"{r['skill']}: \"{r['evidence'][:100]}\" ({r['url']})"
                               for r in missing][:10],
    }


# --- Grader accuracy -----------------------------------------------------------

def ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (Pearson correlation of the ranks); None if undefined."""
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(cov / var, 3) if var else None


def grade_item(item: dict) -> float:
    criteria, _ = nodes.grade(item, item["answer"])
    return nodes.answer_score(item["kind"], criteria, None, 0)


def grader_accuracy(calibration_path: Path = CALIBRATION_PATH, runs: int = REGRADE_RUNS, grade_fn=grade_item) -> dict:
    """Spearman correlation and mean absolute error between grader scores and my labels on the
    calibration set, and the mean score spread when re-grading the same answer `runs` times."""
    if not calibration_path.exists():
        return {"skipped": f"no calibration set at {calibration_path}"}
    items = [json.loads(line) for line in calibration_path.read_text().splitlines() if line.strip()]
    scores = {item["id"]: [grade_fn(item) for _ in range(runs)] for item in items}
    graded = [mean(scores[item["id"]]) for item in items]
    labels = [item["label"] for item in items]
    by_quality: dict[str, list[float]] = {}
    for item, score in zip(items, graded):
        by_quality.setdefault(item.get("quality", "all"), []).append(score)
    return {
        "items": len(items),
        "grades_per_item": runs,
        "spearman_vs_my_labels": spearman(graded, labels),
        "mean_absolute_error": _mean([abs(g - l) for g, l in zip(graded, labels)]),
        "within_0_2_of_label": _ratio(sum(abs(g - l) <= 0.2 for g, l in zip(graded, labels)), len(items)),
        "mean_score_by_quality": {q: _mean(v, 2) for q, v in by_quality.items()},
        "mean_spread_across_regrades": _mean([max(v) - min(v) for v in scores.values()]),
        "disagreements": [f"{item['id']}: grader {g:.2f} vs my label {item['label']:.2f}"
                          for item, g in zip(items, graded) if abs(g - item["label"]) > DISAGREE_BY],
    }


# --- Question quality ----------------------------------------------------------

def judge_question(question: dict, plan_slot: dict | None, gap: str) -> QuestionJudgment:
    llm = config.get_llm().with_structured_output(QuestionJudgment, method="json_schema")
    return llm.invoke(prompts.JUDGE_QUESTION.format(
        skill=question["skill"], kind=question["kind"], difficulty=question["difficulty"], gap=gap,
        question=question["prompt"]))


def question_quality(sessions: list[dict], use_llm: bool = True, judge_fn=judge_question) -> dict:
    """Mean relevance / difficulty fit / clarity (LLM judge); repeated questions; grounded questions;
    coding questions whose tests pass the stored reference solution."""
    questions = [(s, q) for s in sessions for q in s.get("questions", [])]
    prompts_seen = [_norm(q["prompt"]) for _, q in questions]
    repeats = sum(any(SequenceMatcher(None, p, earlier).ratio() >= REPEAT_RATIO for earlier in prompts_seen[:i])
                  for i, p in enumerate(prompts_seen))
    coding = [q for _, q in questions if q["kind"] == "coding" and q.get("tests") and q.get("reference_solution")]
    passing = sum(tools.run_tests(q["reference_solution"], q["tests"])["pass_rate"] == 1.0 for q in coding)
    top_level = [q for _, q in questions if not q.get("parent_id")]
    result = {
        "questions": len(questions),
        "repeated": repeats,
        "grounded_in_posting_or_article": _ratio(sum(bool(q.get("grounding")) for q in top_level), len(top_level)),
        "coding_tests_pass_reference": _ratio(passing, len(coding)),
    }
    if not use_llm:
        return result | {"judge": "skipped (--no-llm)"}
    verdicts, low = [], []
    for session, q in questions:
        slots = session.get("session_plan", [])
        index = int(q["id"].rsplit("q", 1)[-1].rstrip("f")) - 1
        slot = slots[index] if 0 <= index < len(slots) else None
        gaps = nodes.skill_gaps(session.get("skill_demand", {}), session.get("skill_targets", {}),
                                session.get("readiness", {}))
        gap = (f"{q['skill']} (gap {gaps.get(q['skill'], 0):.2f}, readiness "
               f"{session.get('readiness', {}).get(q['skill'], 0):.2f}"
               + (f", chosen as {slot['reason']})" if slot else ")"))
        if q.get("parent_id"):
            gap += "; this is a follow-up to an earlier answer"
        v = judge_fn(q, slot, gap)
        scores = [min(max(v.relevance, 1), 5), min(max(v.difficulty_fit, 1), 5), min(max(v.clarity, 1), 5)]
        verdicts.append(scores)
        if min(scores) <= 2:
            low.append(f"{q['id']} ({q['skill']}, {q['kind']}): relevance {scores[0]}, difficulty {scores[1]}, "
                       f"clarity {scores[2]}")
    return result | {
        "mean_relevance_1_5": _mean([v[0] for v in verdicts], 2),
        "mean_difficulty_fit_1_5": _mean([v[1] for v in verdicts], 2),
        "mean_clarity_1_5": _mean([v[2] for v in verdicts], 2),
        "low_rated_questions": low,
    }


# --- Coaching effect -----------------------------------------------------------

def coaching_effect(conn: sqlite3.Connection) -> dict:
    """Probe follow-ups scoring higher than their parent; how challenge follow-ups went; skills
    with misconceptions that I later answered well (same session or later)."""
    pairs = conn.execute(
        "SELECT pa.score AS parent, fa.score AS follow_up FROM questions f "
        "JOIN assessments fa ON fa.question_id = f.id JOIN assessments pa ON pa.question_id = f.parent_id "
        "WHERE fa.action = 'answered' AND pa.action = 'answered'").fetchall()
    probes = [p for p in pairs if p["parent"] < config.PROBE_BELOW]
    challenges = [p for p in pairs if p["parent"] >= config.CHALLENGE_AT]
    history = conn.execute(
        "SELECT q.skill, a.score, a.misconceptions FROM assessments a JOIN questions q ON q.id = a.question_id "
        "WHERE a.action = 'answered' ORDER BY q.session_date, q.rowid").fetchall()
    with_misconceptions, resolved = 0, 0
    for i, row in enumerate(history):
        if json.loads(row["misconceptions"] or "[]"):
            with_misconceptions += 1
            resolved += any(later["skill"] == row["skill"] and later["score"] >= GOOD_SCORE for later in history[i + 1:])
    return {
        "probe_follow_ups": len(probes),
        "probes_improving_on_parent": _ratio(sum(p["follow_up"] > p["parent"] for p in probes), len(probes)),
        "mean_probe_gain": _mean([p["follow_up"] - p["parent"] for p in probes]),
        "challenge_follow_ups": len(challenges),
        "mean_challenge_score": _mean([p["follow_up"] for p in challenges]),
        "answers_with_misconceptions_later_answered_well": _ratio(resolved, with_misconceptions),
    }


# --- User progress -------------------------------------------------------------

def user_progress(conn: sqlite3.Connection, sessions: list[dict], profile: dict) -> dict:
    """Readiness per assessed skill and per target over sessions; plan tasks done (on time);
    skills that moved from a gap to a strength."""
    trend = memory.skill_trend(conn)
    skills = {s: f"{points[0][1]:.2f} → {points[-1][1]:.2f} ({len(points)} sessions)" for s, points in trend.items()}
    opportunities = memory.stored_opportunities(conn, date.min)   # every open posting, past deadlines too
    by_date: dict[str, dict[str, float]] = {}
    for r in conn.execute("SELECT skill, session_date, readiness FROM skill_history ORDER BY session_date"):
        by_date.setdefault(r["session_date"], {})[r["skill"]] = r["readiness"]
    targets = {}
    for t in profile.get("targets", []):
        values = [nodes.role_readiness(t, opportunities, r)["readiness"] for r in by_date.values()]
        values = [v for v in values if v is not None]
        if values:
            targets[t["role"]] = f"{values[0]:.0%} → {values[-1]:.0%}"
    tasks = conn.execute("SELECT status, due, updated_on FROM plan_tasks").fetchall()
    done = [t for t in tasks if t["status"] == "done"]
    gap_skills, moved = set(), []
    for s in sessions:
        for f in s.get("strengths", []):
            if f["skill"] in gap_skills and f["skill"] not in moved:
                moved.append(f["skill"])
        gap_skills |= {f["skill"] for f in s.get("gaps", [])}
    return {
        "assessed_skills": skills,
        "readiness_by_target": targets,
        "plan_tasks": len(tasks),
        "tasks_done": _ratio(len(done), len(tasks)),
        "tasks_done_on_time": _ratio(sum(t["updated_on"] <= t["due"] for t in done if t["due"] and t["updated_on"]), len(done)),
        "tasks_dropped": sum(t["status"] == "dropped" for t in tasks),
        "gaps_that_became_strengths": moved,
    }


# --- Report grounding ----------------------------------------------------------

def report_grounding(conn: sqlite3.Connection, sessions: list[dict]) -> dict:
    """Findings whose evidence ids exist; links in the reports that come from stored tool results."""
    stored = {r[0] for r in conn.execute("SELECT url FROM opportunities")}
    stored |= {r[0] for r in conn.execute("SELECT resource_url FROM plan_tasks WHERE resource_url IS NOT NULL")}
    findings, cited, links, known, unknown = 0, 0, 0, 0, []
    for s in sessions:
        valid = {q["id"] for q in s.get("questions", [])} | set(s.get("requirement_refs", {}))
        for f in s.get("strengths", []) + s.get("gaps", []):
            findings += 1
            cited += bool(f["evidence"]) and all(e in valid for e in f["evidence"])
        allowed = stored | {r["url"] for r in s.get("resources", [])} | {
            q["grounding"] for q in s.get("questions", []) if q.get("grounding")}
        for url in set(re.findall(r"https?://[^\s<>)|]+", s.get("report_markdown", ""))):
            links += 1
            if url.rstrip(".,") in allowed:
                known += 1
            else:
                unknown.append(url)
    return {
        "findings": findings,
        "findings_with_valid_evidence": _ratio(cited, findings),
        "report_links": links,
        "links_from_stored_tool_results": _ratio(known, links),
        "unknown_links": unknown[:10],
    }


# --- Report --------------------------------------------------------------------

LISTED = {"top_unmapped_skills", "evidence_not_found", "disagreements", "low_rated_questions", "unknown_links"}


def render_report(sections: dict[str, dict], sessions: list[dict]) -> str:
    dates = [s["session_date"].isoformat() for s in sessions]
    lines = [f"# nextLevelPersonalCoachForCareerLevel evaluation — {date.today().isoformat()}", "",
             f"Sessions evaluated: {len(sessions)}" + (f" ({dates[0]} to {dates[-1]})" if dates else ""), ""]
    for title, metrics in sections.items():
        lines += [f"## {title}", "", "| Metric | Value |", "|---|---|"]
        lists = {}
        for name, value in metrics.items():
            if name in LISTED and isinstance(value, list):
                if value:
                    lists[name] = value
                value = len(value)
            if isinstance(value, dict):
                value = ", ".join(f"{k}: {v}" for k, v in value.items()) or "—"
            elif isinstance(value, list):
                value = ", ".join(map(str, value)) or "—"
            lines.append(f"| {name.replace('_', ' ')} | {'—' if value is None else value} |")
        for name, values in lists.items():
            lines += ["", f"**{name.replace('_', ' ').capitalize()}:**", *[f"- {v}" for v in values]]
        lines.append("")
    return "\n".join(lines)


def evaluate(days: int | None = None, use_llm: bool = True, network: bool = True, sessions: list[dict] | None = None,
             judge_fn=judge_question, grade_fn=grade_item, calibration_path: Path = CALIBRATION_PATH) -> str:
    sessions = load_sessions(days) if sessions is None else sessions
    profile = config.load_profile_file()
    today = sessions[-1]["session_date"] if sessions else date.today()
    conn = memory.connect()
    try:
        sections = {
            "Sessions": session_health(sessions),
            "Opportunity accuracy": opportunity_accuracy(
                conn, today, {t["type"] for t in profile.get("targets", [])}, network),
            "Requirement extraction": requirement_extraction(conn),
            "Grader accuracy (calibration set)": (grader_accuracy(calibration_path, grade_fn=grade_fn) if use_llm
                                                  else {"skipped": "--no-llm"}),
            "Question quality": question_quality(sessions, use_llm, judge_fn),
            "Coaching effect": coaching_effect(conn),
            "User progress": user_progress(conn, sessions, profile),
            "Report grounding": report_grounding(conn, sessions),
        }
    finally:
        conn.close()
    return render_report(sections, sessions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate nextLevelPersonalCoachForCareerLevel sessions.")
    parser.add_argument("--days", type=int, default=None, help="only the last N sessions")
    parser.add_argument("--no-llm", action="store_true", help="skip the grader re-runs and question judge")
    parser.add_argument("--no-network", action="store_true", help="skip live posting and link checks")
    args = parser.parse_args()

    report = evaluate(args.days, use_llm=not args.no_llm, network=not args.no_network)
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval_{date.today().isoformat()}.md"
    path.write_text(report)
    print(report)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
