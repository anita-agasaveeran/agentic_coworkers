"""Offline tests for evaluate.py, using sessions saved by the fake-LLM pipeline."""

import json
from datetime import timedelta

import pytest

import evaluate
import memory
from fakes import SESSION_DATE, ScriptedAnswers
from state import QuestionJudgment

EXTRA = "https://example.org/my-posting"


def fake_judge(question, slot, gap):
    return QuestionJudgment(relevance=5, difficulty_fit=4 if question.get("parent_id") else 2, clarity=5)


@pytest.fixture
def sessions(env):
    env["run"](ScriptedAnswers(), SESSION_DATE, session_size=3, extra_postings=[EXTRA])
    env["run"](ScriptedAnswers(text_answers=["strong answer", "weak answer"], check_in={"Read the RAG guide": "done"}),
               SESSION_DATE + timedelta(days=2), session_size=3)
    return evaluate.load_sessions()


def calibration(tmp_path, labels):
    path = tmp_path / "calibration.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"i{i}", "quality": q, "label": label, "answer": "a"})
                              for i, (q, label) in enumerate(labels)))
    return path


def test_load_sessions(sessions):
    assert [s["session_date"] for s in sessions] == [SESSION_DATE, SESSION_DATE + timedelta(days=2)]
    assert len(evaluate.load_sessions(days=1)) == 1


def test_offline_metrics(sessions):
    conn = memory.connect()
    try:
        opps = evaluate.opportunity_accuracy(conn, SESSION_DATE, {"internship", "research"}, network=False)
        extraction = evaluate.requirement_extraction(conn)
        coaching = evaluate.coaching_effect(conn)
        progress = evaluate.user_progress(conn, sessions, {"targets": [
            {"role": "ML engineer intern", "type": "internship"}]})
        grounding = evaluate.report_grounding(conn, sessions)
    finally:
        conn.close()

    assert (opps["analyzed"], opps["kept_open"], opps["dropped_closed"], opps["dropped_not_posting_or_ineligible"]) \
        == (5, 3, 1, 1)
    assert opps["open_matching_a_target_type"] == "3/3 (100%)" and opps["deadlines_in_future"] == "2/2 (100%)"
    assert extraction["evidence_found_in_posting"] == "8/8 (100%)"
    assert extraction["mapped_to_taxonomy"] == "7/8 (88%)" and extraction["top_unmapped_skills"] == ["fortran (1)"]
    assert coaching["probes_improving_on_parent"] == "2/2 (100%)"        # weak 0.25 -> "ok" 0.75, both days
    assert coaching["answers_with_misconceptions_later_answered_well"] == "2/2 (100%)"
    assert progress["tasks_done"] == "1/3 (33%)" and progress["tasks_done_on_time"] == "1/1 (100%)"
    assert "ML engineer intern" in progress["readiness_by_target"]
    assert grounding["findings_with_valid_evidence"] == "4/4 (100%)"
    assert grounding["links_from_stored_tool_results"] == f"{grounding['report_links']}/{grounding['report_links']} (100%)"


def test_question_quality(sessions):
    quality = evaluate.question_quality(sessions, judge_fn=fake_judge)
    assert quality["repeated"] == 0
    assert quality["coding_tests_pass_reference"] == "2/2 (100%)"
    assert quality["mean_relevance_1_5"] == 5.0
    assert len(quality["low_rated_questions"]) == 6                      # top-level questions rated difficulty 2
    assert "judge" in evaluate.question_quality(sessions, use_llm=False)


def test_evaluate_report(sessions, tmp_path):
    path = calibration(tmp_path, [("good", 0.9), ("wrong", 0.1)])
    report = evaluate.evaluate(network=False, judge_fn=fake_judge, grade_fn=lambda item: item["label"],
                               calibration_path=path)
    for heading in ("## Sessions", "## Opportunity accuracy", "## Requirement extraction",
                    "## Grader accuracy (calibration set)", "## Question quality", "## Coaching effect",
                    "## User progress", "## Report grounding"):
        assert heading in report
    assert "Sessions evaluated: 2 (2026-11-01 to 2026-11-03)" in report
    assert "| spearman vs my labels | 1.0 |" in report
    assert "**Low rated questions:**" in report
    assert "skipped" in evaluate.evaluate(use_llm=False, network=False)


def test_grader_accuracy_reports_disagreements_and_spread(tmp_path):
    path = calibration(tmp_path, [("good", 0.9), ("partial", 0.5), ("wrong", 0.1)])
    grades = iter([0.8, 0.9, 0.85, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5])        # 3 re-grades per item
    result = evaluate.grader_accuracy(path, runs=3, grade_fn=lambda item: next(grades))
    assert result["spearman_vs_my_labels"] == 0.5          # graded order good > wrong > partial
    assert result["mean_absolute_error"] == round((0.05 + 0.3 + 0.3) / 3, 3)
    assert result["disagreements"] == ["i1: grader 0.20 vs my label 0.50", "i2: grader 0.40 vs my label 0.10"]
    assert result["mean_spread_across_regrades"] == round((0.1 + 0 + 0.2) / 3, 3)
    assert evaluate.grader_accuracy(tmp_path / "missing.jsonl")["skipped"]


def test_calibration_set_is_well_formed():
    items = [json.loads(line) for line in evaluate.CALIBRATION_PATH.read_text().splitlines()]
    assert len(items) >= 12 and len({i["id"] for i in items}) == len(items)
    for item in items:
        assert {"skill", "kind", "difficulty", "prompt", "reference_answer", "rubric", "answer", "label"} <= set(item)
        assert 0 <= item["label"] <= 1
    by_question: dict[str, dict[str, float]] = {}
    for item in items:
        by_question.setdefault(item["question_id"], {})[item["quality"]] = item["label"]
    assert all(q["good"] > q["partial"] > q["wrong"] for q in by_question.values())


@pytest.mark.parametrize("x, y, expected", [
    ([1, 2, 3], [10, 20, 30], 1.0), ([1, 2, 3], [3, 2, 1], -1.0), ([1, 1, 2], [1, 2, 3], 0.866),
    ([1], [1], None), ([1, 1], [1, 2], None),
])
def test_spearman(x, y, expected):
    assert evaluate.spearman(x, y) == expected


@pytest.mark.parametrize("evidence, expected", [
    ("Experience with  PyTorch", True), ('"experience with pytorch."', True),
    ("Experience with ... Kubernetes", True), ("Experience with TensorFlow", False), ("", False),
])
def test_evidence_found(evidence, expected):
    assert evaluate.evidence_found(evidence, "Requirements:\nExperience with PyTorch and Kubernetes.") is expected


def test_broken_links_count_as_not_open(sessions, env):
    conn = memory.connect()
    try:
        urls = [r[0] for r in conn.execute("SELECT url FROM opportunities WHERE status = 'open' ORDER BY fit_score DESC")]
        env["broken"].add(urls[0])
        opps = evaluate.opportunity_accuracy(conn, SESSION_DATE, {"internship"}, network=True)
    finally:
        conn.close()
    assert opps["links_resolving"] == opps["still_open"] == f"{len(urls) - 1}/{len(urls)} ({(len(urls) - 1) / len(urls):.0%})"
