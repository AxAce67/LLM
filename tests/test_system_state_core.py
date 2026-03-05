import types

import main_controller


class _DummyDB:
    def __init__(self):
        self.set_calls = []
        self.lock_ok = True

    def get_my_target_status(self, node_id):
        return "stop"

    def set_node_target_status(self, node_id, target):
        self.set_calls.append((node_id, target))

    def try_acquire_ha_leader_lock(self, lock_key):
        return self.lock_ok

    def ha_leader_lock_alive(self):
        return True

    def release_ha_leader_lock(self):
        return None


def _make_state_obj():
    # __init__ を通さず、DB接続などの副作用を避ける
    obj = main_controller.SystemState.__new__(main_controller.SystemState)
    obj.is_dashboard = False
    obj.node_id = "node-1234"
    obj.role = "master"
    obj._last_target_check_at = 0.0
    obj.db_manager = _DummyDB()
    obj.ha_enabled = False
    obj.ha_lock_key = 424242
    obj._ha_is_leader = False
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


def test_ha_leader_acquire_success():
    obj = _make_state_obj()
    obj.ha_enabled = True
    obj.state["is_running"] = True
    assert obj.ensure_ha_leader() is True
    assert obj._ha_is_leader is True
    assert obj._node_runtime_status() == "running"


def test_ha_standby_when_lock_not_acquired():
    obj = _make_state_obj()
    obj.ha_enabled = True
    obj.state["is_running"] = True
    obj.db_manager.lock_ok = False
    assert obj.ensure_ha_leader() is False
    assert obj._ha_is_leader is False
    assert obj._node_runtime_status() == "standby"
