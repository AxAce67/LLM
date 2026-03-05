import types

import main_controller


class _DummyDB:
    def __init__(self):
        self.set_calls = []

    def get_my_target_status(self, node_id):
        return "stop"

    def set_node_target_status(self, node_id, target):
        self.set_calls.append((node_id, target))


def _make_state_obj():
    # __init__ を通さず、DB接続などの副作用を避ける
    obj = main_controller.SystemState.__new__(main_controller.SystemState)
    obj.is_dashboard = False
    obj.node_id = "node-1234"
    obj.role = "master"
    obj._last_target_check_at = 0.0
    obj.db_manager = _DummyDB()
    obj.state = {
        "node_id": obj.node_id,
        "role": obj.role,
        "is_running": True,
        "current_phase": "Training",
        "logs": [],
        "stats": {},
        "system": {},
    }
    obj.save = types.MethodType(lambda self: None, obj)
    return obj


def test_merge_loaded_state_keeps_identity_and_fills_defaults():
    obj = _make_state_obj()
    # 壊れた/古いステータス（重要キー欠損）を想定
    loaded = {"node_id": "foreign", "role": "worker", "stats": {"current_epoch": 12}}
    merged = obj._merge_loaded_state(loaded)
    assert merged["node_id"] == "node-1234"
    assert merged["role"] == "master"
    assert "is_running" in merged
    assert "stats" in merged and "current_epoch" in merged["stats"]


def test_should_stop_now_consumes_remote_stop_signal():
    obj = _make_state_obj()
    assert obj.should_stop_now(read_remote=True) is True
    assert obj.state["is_running"] is False
    assert obj.state["current_phase"] == "Idle"
    assert obj.db_manager.set_calls[-1] == ("node-1234", "unspecified")
