"""
Phase D — Diffusion Policy runner for learned motor control.

Loads the trained DP model and provides a simple inference interface:
    runner = DPRunner()
    runner.load_checkpoint("/path/to/checkpoint/002000/pretrained_model", stats_path=".../meta/stats.json")
    action = runner.select_action(frame_rgb, state_dict)

The model handles n_obs_steps caching internally via select_action.
Caller repeats at ~20Hz in a closed loop.
The output is unnormalized from MIN_MAX to real joint values.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class DPRunner:
    """Diffusion Policy inference wrapper for real-robot control."""

    def __init__(self, device: str | None = None):
        self.policy = None
        self.config = None
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._loaded = False
        # MIN_MAX normalization stats for action unnormalization
        self._action_min: np.ndarray | None = None
        self._action_max: np.ndarray | None = None

    def load_checkpoint(self, checkpoint_dir: str, stats_path: str | None = None) -> bool:
        """
        Load a trained DP checkpoint.

        Args:
            checkpoint_dir: Path containing config.json + model.safetensors.
            stats_path: Path to dataset stats.json for action unnormalization.
                        If None, tries checkpoint_dir/../training_state/... and
                        common dataset paths.
        """
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        try:
            self.policy = DiffusionPolicy.from_pretrained(checkpoint_dir)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
            self.config = self.policy.config
            self._loaded = True
            logger.info(
                f"DPRunner ready on {self.device}: "
                f"horizon={self.config.horizon}, n_obs={self.config.n_obs_steps}, "
                f"n_action={self.config.n_action_steps}"
            )
        except Exception as e:
            logger.error(f"Failed to load DP checkpoint: {e}")
            return False

        # Load normalization stats for action unnormalization
        if stats_path is None:
            # Auto-discover: look in common locations relative to checkpoint
            candidates = [
                Path(checkpoint_dir).parent.parent.parent.parent
                / "datasets/local/zhenbang_pickplace_dp/meta/stats.json",
                Path(checkpoint_dir).parent.parent.parent / "training_state/training_state.json",
            ]
            for c in candidates:
                if c.exists():
                    stats_path = str(c)
                    break

        if stats_path and Path(stats_path).exists():
            try:
                stats_data = json.loads(Path(stats_path).read_text())
                # stats might be at the top level or nested under "action"
                action_stats = stats_data.get("action", stats_data)
                self._action_min = np.array(action_stats["min"], dtype=np.float32)
                self._action_max = np.array(action_stats["max"], dtype=np.float32)
                # Override gripper max: training data only goes to 62 (leader
                # range), but the actual servo can reach 100.  Without this,
                # the DP never commands the gripper tight enough to hold.
                self._action_max[5] = 100.0
                logger.info(
                    f"Loaded action stats: min={self._action_min.tolist()}, max={self._action_max.tolist()}"
                    f" (gripper max overridden to 100.0)"
                )
            except Exception as e:
                logger.warning(f"Failed to load stats from {stats_path}: {e}")

        return True

    def reset(self):
        """Reset internal observation/action queues before a new episode."""
        if self.policy is not None:
            self.policy.reset()

    def _unnormalize(self, action_np: np.ndarray) -> np.ndarray:
        """MIN_MAX unnormalize from [-1,1] (model output) to real joint values."""
        if self._action_min is not None and self._action_max is not None:
            # MIN_MAX in LeRobot: norm = (raw - min) / (max - min), range [0, 1]
            # Model outputs in roughly [-1, 1], so convert: norm_model * 0.5 + 0.5
            norm_01 = (action_np.astype(np.float32) + 1.0) * 0.5
            clip = np.clip(norm_01, 0.0, 1.0)
            return clip * (self._action_max - self._action_min) + self._action_min
        return action_np

    @torch.no_grad()
    def select_action(self, frame: np.ndarray, state: dict[str, float]) -> dict[str, float] | None:
        """
        Given current camera frame + joint state, return next joint target.

        Args:
            frame: (H, W, 3) RGB array from wrist cam.
            state: dict of {joint_name: value}.

        Returns:
            Action dict with {f"{j}.pos": value} in REAL joint units.
        """
        if not self._loaded or self.policy is None:
            return None

        try:
            import cv2
        except ImportError:
            logger.error("cv2 required for DP resize")
            return None

        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

        state_vec = np.array(
            [float(state.get(j, 0.0)) for j in JOINT_NAMES],
            dtype=np.float32,
        )

        batch = {
            "observation.state": (torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)),
            "observation.images.camera1": (
                torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            ),
        }

        try:
            action_tensor = self.policy.select_action(batch)
            if action_tensor.ndim == 2:
                action_tensor = action_tensor.squeeze(0)
            action_np = action_tensor.cpu().numpy().astype(np.float32)
            action_np = self._unnormalize(action_np)
            return {f"{j}.pos": float(action_np[i]) for i, j in enumerate(JOINT_NAMES)}
        except Exception as e:
            logger.warning(f"DP select_action error: {e}")
            return None
