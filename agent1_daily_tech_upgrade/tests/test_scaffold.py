from agent import build_graph
from config import TOP_K, MAX_RETRIES, SCORE_WEIGHTS, load_profile_file, load_sources
from memory import connect
from nodes import route_after_scoring

EXPECTED_NODES = {
    "load_profile", "plan_queries", "fetch_candidates", "dedupe_and_filter", "score_and_rank",
    "broaden_queries", "extract_full_text", "summarize_and_personalize", "compose_brief",
    "collect_feedback", "update_memory",
}


def test_graph_compiles_with_all_nodes():
    graph = build_graph().get_graph()
    assert EXPECTED_NODES <= set(graph.nodes)


def test_route_retries_until_top_k_or_max_retries():
    assert route_after_scoring({"selected": [{}] * (TOP_K - 1), "retries": 0}) == "broaden_queries"
    assert route_after_scoring({"selected": [{}] * TOP_K, "retries": 0}) == "extract_full_text"
    assert route_after_scoring({"selected": [], "retries": MAX_RETRIES}) == "extract_full_text"


def test_score_weights_sum_to_one():
    assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def test_config_files_load():
    assert load_profile_file()["interests"]
    assert load_sources()["sources"]


def test_history_schema(tmp_path):
    conn = connect(tmp_path / "history.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"articles_seen", "feedback", "topic_progress"} <= tables
