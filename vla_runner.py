"""
SmolVLA inference wrapper for Phase H.
Loads the SFT'd checkpoint, builds observations, returns action commands.

Usage:
    runner = VLARunner()             # loads ./checkpoints/last by default
    runner.reset()                   # at start of each episode
    while episode_active:
        action = runner.get_action(
            frame_bgr,               # (H, W, 3) BGR from camera
            joint_state_dict,        # {shoulder_pan: ..., gripper: ...}
            instruction="pick up the carrot",
        )
        robot.go_to_pose(action, ...)
"""
from __future__ import annotations
import logging
import threading
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import cv2
import torch

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

DEFAULT_CHECKPOINT = (
    "/Users/chenyuying/Downloads/lerobot_repo/checkpoints/"
    "smolvla_zhenbang/checkpoints/last/pretrained_model"
)


# Map our voice/UI inputs to the English instructions seen during training.
# Free-text targets fall through unchanged.
INSTRUCTION_MAP = {
    # Default carrot routing → 35 demo set is RIGHT-to-LEFT pick-and-place
    "萝卜":   "pick up the carrot and place it on the left",
    "carrot": "pick up the carrot and place it on the left",
    # If you want pure pick (no place), use "拿起萝卜" or "carrot pick"
    "拿起萝卜": "pick up the carrot",
    "carrot pick": "pick up the carrot",
    "纸巾":   "pull a tissue from the roll",
    "tissue": "pull a tissue from the roll",
    "可乐":   "push the cola can sideways",
    "cola":   "push the cola can sideways",
}


def map_target_to_instruction(target_text: str) -> str:
    """Resolve a UI/voice target string to a SmolVLA instruction."""
    if not target_text:
        return ""
    key = target_text.strip()
    if key in INSTRUCTION_MAP:
        return INSTRUCTION_MAP[key]
    # heuristic match
    low = key.lower()
    for k, v in INSTRUCTION_MAP.items():
        if k in key or k.lower() in low:
            return v
    # fall back: use as-is, model may or may not understand
    return key


class VLARunner:
    """Inference wrapper around the SFT'd SmolVLA policy."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str = "mps",
    ):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Did you run training? Expected path:\n"
                f"  {DEFAULT_CHECKPOINT}"
            )
        self.device = device

        logger.info(f"Loading SmolVLA from {self.checkpoint_path}")
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint_path)
        self.policy.eval()
        self.policy.to(device)

        # Resolve tokenizer (used for instruction encoding)
        self.tokenizer = self._resolve_tokenizer()

        # IMPORTANT: SmolVLAPolicy.select_action() returns NORMALIZED action
        # (z-score scaled), NOT real joint angles. We have to manually load
        # action stats from the postprocessor safetensors and unnormalize.
        # Same for STATE input — model expects normalized state.
        self.action_mean, self.action_std = self._load_action_stats()
        self.state_mean, self.state_std = self._load_state_stats()
        logger.info("Loaded normalization stats: "
                    "action_mean=%s ... action_std=%s",
                    self.action_mean[:3].tolist(), self.action_std[:3].tolist())

        # EMA action smoothing — reduces shake from state==action bug
        self.action_ema_alpha: float = 0.5
        self._prev_action: Optional[np.ndarray] = None

        # Manual joint offsets — to compensate for the model's systematic
        # depth-estimation errors when picking. Increase lift_offset (positive)
        # to make the arm descend MORE. Default 0 (no offset).
        # Indexed: [pan, lift, elbow, wrist_flex, wrist_roll, gripper]
        self.action_offset: np.ndarray = np.zeros(6, dtype=np.float32)

        # Single-thread the GPU access — game loop and any UI thread share this.
        self._lock = threading.Lock()
        logger.info("VLARunner ready (device=%s, ema_alpha=%.2f)",
                    device, self.action_ema_alpha)

    # ------------------------------------------------------------------
    def _load_action_stats(self):
        """Action mean/std from policy_postprocessor_*.safetensors."""
        from safetensors.torch import load_file
        stats_file = (self.checkpoint_path /
                      "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        if not stats_file.exists():
            logger.warning("No postprocessor stats; actions will be raw normalized")
            return np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)
        stats = load_file(str(stats_file))
        return (stats["action.mean"].cpu().numpy().astype(np.float32),
                stats["action.std"].cpu().numpy().astype(np.float32))

    def _load_state_stats(self):
        """State mean/std from policy_preprocessor_*.safetensors."""
        from safetensors.torch import load_file
        stats_file = (self.checkpoint_path /
                      "policy_preprocessor_step_5_normalizer_processor.safetensors")
        if not stats_file.exists():
            logger.warning("No preprocessor stats; state will be raw")
            return np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)
        stats = load_file(str(stats_file))
        return (stats["observation.state.mean"].cpu().numpy().astype(np.float32),
                stats["observation.state.std"].cpu().numpy().astype(np.float32))

    # ------------------------------------------------------------------
    def _resolve_tokenizer(self):
        # Try common paths inside SmolVLA
        for path in [
            ("model", "vlm_with_expert", "processor", "tokenizer"),
            ("vlm_with_expert", "processor", "tokenizer"),
            ("model", "vlm_with_expert", "tokenizer"),
        ]:
            obj = self.policy
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok and callable(obj):
                logger.info("tokenizer found at policy.%s", ".".join(path))
                return obj
        # Fallback to HuggingFace
        from transformers import AutoTokenizer
        logger.info("tokenizer fallback → AutoTokenizer(SmolVLM2-500M)")
        return AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # ------------------------------------------------------------------
    def reset(self):
        """Clear internal action-chunk cache. Call at start of each episode."""
        with self._lock:
            self.policy.reset()
            self._prev_action = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(
        self,
        frame_bgr: np.ndarray,
        joint_state: Dict[str, float],
        instruction: str,
    ) -> Dict[str, float]:
        """
        Run policy forward, return next action as joint dict.
        SmolVLA caches an action chunk internally — only every chunk_size calls
        does this trigger a heavy forward pass (~1s on MPS).
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        with self._lock:
            obs = self._build_observation(frame_bgr, joint_state, instruction)
            action = self.policy.select_action(obs)
        action_np = action.detach().cpu().float().numpy().reshape(-1)
        if action_np.shape[0] != 6:
            raise RuntimeError(
                f"Unexpected action shape {action_np.shape}, expected (6,)")
        # Unnormalize: model returns z-score; bring back to real joint angles.
        action_np = action_np * self.action_std + self.action_mean
        # EMA smoothing — reduces high-freq jitter from state-action mismatch
        if self._prev_action is None:
            self._prev_action = action_np.copy()
        else:
            a = self.action_ema_alpha
            action_np = a * action_np + (1.0 - a) * self._prev_action
            self._prev_action = action_np.copy()
        # Apply manual joint offset (e.g. compensate for depth-perception bias).
        # NOTE: applied AFTER EMA so it doesn't get smoothed out / propagated.
        action_np = action_np + self.action_offset
        return dict(zip(JOINT_NAMES, action_np.tolist()))

    # ------------------------------------------------------------------
    def _build_observation(
        self,
        frame_bgr: np.ndarray,
        joint_state: Dict[str, float],
        instruction: str,
    ) -> Dict:
        # Image: BGR → RGB → resize 256×256 → CHW float [0,1]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256))
        img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # State: normalize via z-score (matches what training data loader did)
        state_vec = np.array(
            [float(joint_state.get(j, 0.0)) for j in JOINT_NAMES],
            dtype=np.float32,
        )
        state_norm = (state_vec - self.state_mean) / np.maximum(self.state_std, 1e-8)
        state = torch.from_numpy(state_norm).unsqueeze(0)

        # Language tokens
        toks = self.tokenizer([instruction], return_tensors="pt", padding=True)

        obs = {
            # Replicate single camera to all 3 slots — empirically gives
            # better predictions than zeros (especially for gripper timing).
            # Diagnosed via debug_vla_v2.py: at gripping moment, zeros made
            # model predict grip=5 (open) when truth was 47 (closed).
            "observation.images.camera1": img,
            "observation.images.camera2": img.clone(),
            "observation.images.camera3": img.clone(),
            "observation.state": state,
            "observation.language.tokens": toks["input_ids"].long(),
            "observation.language.attention_mask": toks["attention_mask"].bool(),
        }
        # Move to model device
        for k, v in obs.items():
            if hasattr(v, "to"):
                obs[k] = v.to(self.device)
        return obs

    # ------------------------------------------------------------------
    def __repr__(self):
        return f"VLARunner(checkpoint={self.checkpoint_path}, device={self.device})"
