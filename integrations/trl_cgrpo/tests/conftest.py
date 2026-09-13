import pytest


@pytest.fixture(autouse=True)
def force_use_cpu_without_accelerator(monkeypatch):
    """Mirror TRL's own test fixture: TRL configs default `bf16=True`, which fails validation on a CPU-only machine.

    On a machine with an accelerator this is a no-op.
    """
    from transformers.testing_utils import torch_device

    if torch_device is not None and torch_device != "cpu":
        return

    try:
        from trl.trainer.base_config import _BaseConfig as config_cls
    except ImportError:  # older TRL without a shared base config
        from trl import GRPOConfig as config_cls

    original_post_init = config_cls.__post_init__

    def patched_post_init(self):
        self.use_cpu = True
        original_post_init(self)

    monkeypatch.setattr(config_cls, "__post_init__", patched_post_init)
