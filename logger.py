import os
import json
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

class JsonlLogger:
    """
    Always-on local logger that writes one JSON per line.
    Safe, simple, and works everywhere.
    """
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, "events.jsonl")
        self._f = open(self.path, "a", buffering=1)

    def log(self, step: int, data: Dict[str, Any]):
        rec = {"ts": time.time(), "step": int(step), **data}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

class WandbLogger:
    def __init__(self, project: str, run_name: str, config: Dict[str, Any], log_dir: str):
        import wandb
        os.makedirs(log_dir, exist_ok=True)
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config, dir=log_dir)

    def log(self, step: int, data: Dict[str, Any]):
        self.wandb.log(data, step=step)

    def log_table(self, name: str, columns: List[str], rows: List[List[Any]], step: int):
        table = self.wandb.Table(columns=columns, data=rows)
        self.wandb.log({name: table}, step=step)

    def finish(self):
        self.run.finish()

class TensorboardLogger:
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.w = SummaryWriter(log_dir=log_dir)

    def log(self, step: int, data: Dict[str, Any]):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                self.w.add_scalar(k, v, global_step=step)

    def flush(self):
        self.w.flush()

def make_loggers(
    cfg,
    log_dir: str,
    use_wandb: bool = True,
    wandb_project: str = "conformal-grpo-rlvr",
    run_name: Optional[str] = None,
    use_tensorboard: bool = True,
):
    os.makedirs(log_dir, exist_ok=True)
    jsonl = JsonlLogger(log_dir)

    wb = None
    if use_wandb:
        try:
            wb = WandbLogger(wandb_project, run_name or "run", asdict(cfg), log_dir)
        except Exception as e:
            print(f"[WARN] W&B init failed, continuing without it: {e}")

    tb = None
    if use_tensorboard:
        try:
            tb = TensorboardLogger(os.path.join(log_dir, "tb"))
        except Exception as e:
            print(f"[WARN] TensorBoard init failed, continuing without it: {e}")

    return jsonl, wb, tb
