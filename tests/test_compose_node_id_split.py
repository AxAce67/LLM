from pathlib import Path


def test_compose_has_separate_node_id_files_for_services():
    compose = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    text = compose.read_text(encoding="utf-8")
    assert "NODE_ID_FILE=/app/checkpoints/node_id_engine.txt" in text
    assert "NODE_ID_FILE=/app/checkpoints/node_id_dashboard.txt" in text
