"""
GraspMemory — 把成功的抓取偏移量持久化到 JSON, 下次启动直接复用.

回答用户问题: "有什么记忆机制?"

数据结构 (~/grasp_memory.json):
{
  "carrot": {
    "x_offset": -0.09,
    "y_offset": 0.0,
    "successes": 3,
    "failures": 2,
    "last_success_ts": "2026-05-12T02:24:33"
  },
  "tissue": {...}
}

写入时机:
  - grasp_with_retry 成功 → 把当前 offset 作为最优值 (running mean)
  - grasp_with_retry 失败 → failures += 1

读取时机:
  - primitives 初始化 → 如果有记录, 用记录的 offset 作为默认值
  - UI 可以读 successes / failures 显示统计
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class GraspMemory:
    """Per-target grasp offset memory, persisted to JSON."""

    def __init__(self, path: str = "grasp_memory.json"):
        self.path = Path(path)
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info("GraspMemory loaded %d targets from %s",
                            len(d), self.path)
                return d
            except Exception as e:
                logger.warning("Failed to read %s: %s", self.path, e)
        return {}

    def _save(self):
        try:
            self.path.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save %s: %s", self.path, e)

    # ------------------------------------------------------------------
    def get_offset(self, target_name: str) -> Optional[Tuple[float, float]]:
        """Return (x_offset, y_offset) for target, or None if no memory."""
        rec = self.data.get(target_name)
        if rec is None:
            return None
        return (float(rec.get("x_offset", 0.0)),
                float(rec.get("y_offset", 0.0)))

    def get_stats(self, target_name: str) -> dict:
        """Return success/failure stats for target."""
        return self.data.get(target_name, {
            "successes": 0, "failures": 0, "x_offset": 0.0, "y_offset": 0.0,
        })

    # ------------------------------------------------------------------
    def record_success(self, target_name: str,
                        x_offset: float, y_offset: float):
        """
        Update the stored offset with a running mean of successful offsets.
        Heavier weight on first few — stabilizes quickly.
        """
        rec = self.data.setdefault(target_name, {
            "x_offset": 0.0, "y_offset": 0.0,
            "successes": 0, "failures": 0,
        })
        s = int(rec.get("successes", 0))
        rec["x_offset"] = (rec["x_offset"] * s + x_offset) / (s + 1)
        rec["y_offset"] = (rec["y_offset"] * s + y_offset) / (s + 1)
        rec["successes"] = s + 1
        rec["last_success_ts"] = datetime.now().isoformat()
        self._save()
        logger.info("GraspMemory: %s success #%d, offset=(%.3f, %.3f)",
                    target_name, rec["successes"], rec["x_offset"], rec["y_offset"])

    def record_failure(self, target_name: str):
        rec = self.data.setdefault(target_name, {
            "x_offset": 0.0, "y_offset": 0.0,
            "successes": 0, "failures": 0,
        })
        rec["failures"] = int(rec.get("failures", 0)) + 1
        rec["last_failure_ts"] = datetime.now().isoformat()
        self._save()

    # ------------------------------------------------------------------
    def __repr__(self):
        n = len(self.data)
        return f"GraspMemory(path={self.path}, targets={n})"
