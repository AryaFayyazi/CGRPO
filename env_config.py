import os
import pathlib

def configure_hf_caches(preferred_root: str = os.path.expanduser("~/.cache/huggingface"), datasets_root: str = os.path.expanduser("~/.cache/huggingface/datasets")):
    """
    Writable HF caches under ~/.cache/huggingface.
    Returns: (HF_DATASETS_CACHE, TRANSFORMERS_CACHE)
    """
    root = os.path.expanduser(preferred_root)
    hub = os.path.join(root, "hub")
    ds  = os.path.abspath(datasets_root)
    tfm = os.path.join(root, "transformers")

    for d in (root, hub, ds, tfm):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    env = {
        "HF_HOME": root,
        "HF_HUB_CACHE": hub,
        "HF_DATASETS_CACHE": ds,
        "TRANSFORMERS_CACHE": tfm,
    }

    for k, v in env.items():
        if not os.environ.get(k):
            os.environ[k] = v

    os.environ.setdefault("HF_DATASETS_IN_MEMORY_MAX_SIZE", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return os.environ["HF_DATASETS_CACHE"], os.environ["TRANSFORMERS_CACHE"]

__all__ = ["configure_hf_caches"]
