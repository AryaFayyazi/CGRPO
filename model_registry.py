import os
from typing import Dict

# Paste your registry here verbatim
# Local snapshot root. Override with CGRPO_MODEL_ROOT, or replace the
# "hf_id" values below with plain Hugging Face repo ids to download on demand.
_MODEL_ROOT = os.environ.get("CGRPO_MODEL_ROOT", "/data/models")

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # Small
    "qwen2.5-3b": {
        "hf_id": _MODEL_ROOT + "/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
        "group": "small",
        "dtype": "bfloat16",
    },
    "phi-4-mini-instruct": {
        "hf_id": _MODEL_ROOT + "/models--microsoft--Phi-4-mini-instruct/snapshots/5a149550068a1eb93398160d8953f5f56c3603e9",
        "group": "small",
        "dtype": "bfloat16",
    },
    "phi-3.5-mini-instruct": {
        "hf_id": _MODEL_ROOT + "/models--microsoft--Phi-3.5-mini-instruct/snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77",
        "group": "small",
        "dtype": "bfloat16",
    },
    "gemma3_4b": {
        "hf_id": _MODEL_ROOT + "/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
        "group": "small",
        "dtype": "bfloat16",
    },
    # Medium
    "llama3.1-8b": {
        "hf_id": _MODEL_ROOT + "/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        "group": "medium",
        "dtype": "bfloat16",
    },
    "qwen2.5-math-7b-instruct": {
        "hf_id": _MODEL_ROOT + "/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d",
        "group": "medium",
        "dtype": "bfloat16",
    },
    "QWEN2.5-7b": {
        "hf_id": _MODEL_ROOT + "/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "group": "medium",
        "dtype": "bfloat16",
    },
    "Qwen3-32b": {
        "hf_id": _MODEL_ROOT + "/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137",
        "group": "medium",
        "dtype": "bfloat16",
    },
    "dolphin-34b": {
        "hf_id": _MODEL_ROOT + "/models--dphn--dolphin-2.9.1-yi-1.5-34b/snapshots/0141cba238d0faad09bc240ea7af14c9ea5aec44",
        "group": "medium",
        "dtype": "bfloat16",
    },
    # Large
    "llama-70b": {
        "hf_id": _MODEL_ROOT + "/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "group": "large",
        "dtype": "bfloat16",
    },
}

def get_model_path(model_key: str) -> str:
    if model_key not in MODEL_REGISTRY:
        keys = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model_key='{model_key}'. Available: {keys}")
    return MODEL_REGISTRY[model_key]["hf_id"]

def get_model_dtype_str(model_key: str) -> str:
    return MODEL_REGISTRY[model_key].get("dtype", "bfloat16")
