from core_llm.config import ModelConfig, TrainConfig
from core_llm.train.loop import resolve_amp, resolve_batch_size


def test_resolve_amp_auto_enables_cuda_only():
    assert resolve_amp(TrainConfig(amp="auto"), "cuda") is True
    assert resolve_amp(TrainConfig(amp="auto"), "cpu") is False


def test_resolve_amp_false_disables_amp():
    assert resolve_amp(TrainConfig(amp=False), "cuda") is False


def test_resolve_batch_size_respects_explicit_value():
    train_config = TrainConfig(batch_size=6)
    model_config = ModelConfig()
    assert resolve_batch_size(train_config, model_config, "cuda") == 6


def test_resolve_batch_size_falls_back_to_one_on_cpu():
    train_config = TrainConfig(batch_size=0)
    model_config = ModelConfig()
    assert resolve_batch_size(train_config, model_config, "cpu") == 1
