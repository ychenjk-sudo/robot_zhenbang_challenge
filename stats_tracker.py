"""
Lightweight stats tracker — replaces REINFORCETrainer for the case where
we only need to track accuracy/reward stats (no neural network needed).

Phase H 时代 SmolVLA 自己负责动作生成, 不需要 CLIP, 不需要 policy 网络,
trainer 只是个数字桶. 保留同样的接口让 game.py / main.py 不用改.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class TrackerConfig:
    accuracy_window: int = 20
    baseline_decay: float = 0.9
    # Kept for legacy compat (detector.py reads target_prompts at init)
    target_prompts: tuple = (
        "a fresh orange carrot",
        "a roll of white toilet paper",
        "a red can of Coca-Cola",
    )


class StatsTracker:
    """Drop-in replacement for REINFORCETrainer (stats-only)."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()
        self.baseline: float = 0.0
        self.episode_count: int = 0
        self.total_reward: float = 0.0
        self.reward_history: List[float] = []
        self.loss_history: List[float] = []          # always 0 (no learning)
        self.prob_history: List[float] = []          # always 0
        self.accuracy_window = deque(maxlen=self.cfg.accuracy_window)
        # Stale fields kept so game.py / main.py keep compiling
        self.current_eps: float = 0.0

    @property
    def current_accuracy(self) -> float:
        if not self.accuracy_window:
            return 0.0
        return float(np.mean(self.accuracy_window))

    def update(self, frame, target_idx: int, action_idx: int,
               log_prob_val: float, reward: float):
        """Just record reward stats; no model update."""
        self.episode_count += 1
        self.total_reward += float(reward)
        self.reward_history.append(float(reward))
        self.accuracy_window.append(1 if reward > 0.5 else 0)
        self.loss_history.append(0.0)
        self.prob_history.append(0.0)
        self.baseline = (
            self.cfg.baseline_decay * self.baseline
            + (1.0 - self.cfg.baseline_decay) * float(reward)
        )

    def reset_stats(self):
        """Clear all accumulated stats."""
        self.baseline = 0.0
        self.episode_count = 0
        self.total_reward = 0.0
        self.reward_history.clear()
        self.loss_history.clear()
        self.prob_history.clear()
        self.accuracy_window.clear()
