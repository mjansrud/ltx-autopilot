"""
Persistent pipeline state — survives crashes and restarts.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class PipelineState:
    def __init__(self, state_file: str | Path):
        self.path = Path(state_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            log.info("Loaded state: batch=%d, total_steps=%d", data["batch_num"], data["total_steps"])
            return data
        return {
            "batch_num": 0,
            "total_steps": 0,
            "last_checkpoint": None,
            "last_batch_time": None,
            "batches_since_base_eval": 0,
            "history": [],
        }

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    @property
    def batch_num(self) -> int:
        return self.data["batch_num"]

    @property
    def total_steps(self) -> int:
        return self.data["total_steps"]

    @property
    def last_checkpoint(self) -> str | None:
        return self.data.get("last_checkpoint")

    @last_checkpoint.setter
    def last_checkpoint(self, value: str | None):
        self.data["last_checkpoint"] = value

    def advance_batch(self, steps_trained: int, checkpoint_path: str | None = None):
        self.data["batch_num"] += 1
        self.data["total_steps"] += steps_trained
        self.data["batches_since_base_eval"] += 1
        self.data["last_batch_time"] = datetime.now(timezone.utc).isoformat()
        if checkpoint_path:
            self.data["last_checkpoint"] = checkpoint_path
        self.data["history"].append({
            "batch": self.data["batch_num"],
            "steps": steps_trained,
            "total_steps": self.data["total_steps"],
            "checkpoint": checkpoint_path,
            "time": self.data["last_batch_time"],
        })
        # Keep only last 100 history entries
        self.data["history"] = self.data["history"][-100:]
        self.save()

    def should_compare_base(self, every_n: int) -> bool:
        if self.data["batches_since_base_eval"] >= every_n:
            self.data["batches_since_base_eval"] = 0
            self.save()
            return True
        return False
