"""
Learner Module
Online REINFORCE policy gradient for the visual decision task.
Robot learns to map camera observations -> left/right actions via real owner feedback.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import json
from typing import Tuple, Optional, Dict, List
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    lr: float = 1e-3
    gamma: float = 0.99            # discount (unused in single-step but kept for compatibility)
    baseline_decay: float = 0.9    # EMA decay for reward baseline
    entropy_coef: float = 0.01     # encourage exploration early on
    epsilon_start: float = 0.4     # initial epsilon-greedy
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100  # over how many episodes to decay
    image_size: int = 224          # CLIP expects 224×224
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    # CLIP zero-shot vocabulary. Index 0/1 are pickable as game targets via the
    # UI radio (萝卜 / 纸巾). Index 2+ are recognition-only — they show up in the
    # thinking panel so the user can see what CLIP thinks each side is, but the
    # owner can't ask the robot to "tap the cola".
    target_prompts: Tuple[str, ...] = (
        "a fresh orange carrot",
        "a roll of white toilet paper",
        "a red can of Coca-Cola",
    )


# CLIP normalization constants (matches what CLIP was trained on)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VisualPolicy(nn.Module):
    """
    CLIP-backed target-conditional policy.

    Image goes through frozen CLIP image encoder → 512-d embedding.
    Target prompts ("a photo of a carrot" / "...tissue paper") are embedded
    once at init via CLIP text encoder. The current target's text embedding
    is selected (via target onehot @ text_features), concatenated with the
    image embedding, and passed through a small trainable MLP head.

    Only the head's ~260k params are trained; CLIP's ~150M are frozen.
    """
    def __init__(
        self,
        image_size: int = 224,
        num_targets: int = 2,
        output_dim: int = 2,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        target_prompts=("a photo of a carrot", "a photo of tissue paper"),
    ):
        super().__init__()
        import open_clip
        self.image_size = image_size
        self.num_targets = num_targets
        if len(target_prompts) != num_targets:
            raise ValueError(
                f"target_prompts has {len(target_prompts)} entries, expected {num_targets}"
            )

        clip_model_obj, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        # Freeze backbone
        for p in clip_model_obj.parameters():
            p.requires_grad = False
        clip_model_obj.eval()
        self.clip_model = clip_model_obj
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Pre-compute & cache text features (one per target). Done in fp32 for stability.
        with torch.no_grad():
            tokens = self.tokenizer(list(target_prompts))
            text_features = clip_model_obj.encode_text(tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Save as buffer so it moves with .to(device) and is included in state_dict
        self.register_buffer("text_features", text_features)  # (num_targets, embed_dim)

        # Learnable temperature for converting cosine similarity → logit scale.
        # init to log(20) ≈ 3.0, similar to CLIP's own logit_scale init.
        self.logit_scale = nn.Parameter(torch.tensor(3.0))

        # CLIP normalization, registered as buffer for device mobility
        self.register_buffer(
            "_clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x in [0,1] → CLIP-normalized."""
        return (x - self._clip_mean) / self._clip_std

    def _encode_half(self, half: torch.Tensor) -> torch.Tensor:
        """Resize a (B, 3, H, W/2) crop to 224×224 and run through CLIP."""
        if half.shape[2] != 224 or half.shape[3] != 224:
            half = F.interpolate(half, size=(224, 224), mode="bilinear", align_corners=False)
        half = self._normalize(half)
        with torch.no_grad():
            feat = self.clip_model.encode_image(half).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat  # (B, D)

    def forward(self, x: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) RGB images in [0,1] (square, but H/W can vary)
        target_onehot: (B, num_targets) float onehot
        returns: (B, 2) logits
        """
        # Split into left/right halves along width
        W = x.shape[3]
        mid = W // 2
        left_half = x[:, :, :, :mid]
        right_half = x[:, :, :, mid:]

        # Encode each half through CLIP
        l_feat = self._encode_half(left_half)
        r_feat = self._encode_half(right_half)

        # Target text embedding (already normalized in __init__)
        tgt_feat = target_onehot.float() @ self.text_features  # (B, D)
        tgt_feat = tgt_feat / tgt_feat.norm(dim=-1, keepdim=True)

        # Cosine similarities → directly become logits after temperature scaling
        sim_left = (l_feat * tgt_feat).sum(dim=1)   # (B,)
        sim_right = (r_feat * tgt_feat).sum(dim=1)  # (B,)

        scale = self.logit_scale.exp()
        logits = torch.stack([sim_left, sim_right], dim=1) * scale
        return logits

    def trainable_parameters(self):
        """Only the logit temperature is trainable (CLIP backbone frozen, no MLP head)."""
        return [self.logit_scale]


class REINFORCETrainer:
    """
    Online trainer: each episode is one interaction (look -> choose -> reward -> update).
    No replay buffer, no batching: pure online single-sample REINFORCE with EMA baseline.
    """

    def __init__(self, config: Optional[LearningConfig] = None, device: Optional[str] = None):
        self.cfg = config or LearningConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer device: %s", self.device)

        self.num_targets = len(self.cfg.target_prompts)
        self.policy = VisualPolicy(
            image_size=self.cfg.image_size,
            num_targets=self.num_targets,
            clip_model=self.cfg.clip_model,
            clip_pretrained=self.cfg.clip_pretrained,
            target_prompts=self.cfg.target_prompts,
        ).to(self.device)
        # Only train the small head — CLIP backbone is frozen
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-4,
        )

        self.baseline = 0.0
        self.episode_count = 0
        self.total_reward = 0.0
        self.correct_count = 0

        # History for UI visualization
        self.reward_history: deque = deque(maxlen=200)
        self.accuracy_window: deque = deque(maxlen=20)  # sliding window accuracy
        self.loss_history: deque = deque(maxlen=100)
        self.prob_history: deque = deque(maxlen=50)     # log chosen prob

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        OpenCV BGR frame -> RGB tensor in [0,1], shape (1, 3, H, W).
        CLIP-specific normalization is applied inside VisualPolicy.forward,
        so we keep this preprocess in plain [0,1] for visibility/debug.
        """
        size = self.cfg.image_size
        img = cv2.resize(frame_bgr, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def _target_tensor(self, target_idx: int) -> torch.Tensor:
        """Build (1, num_targets) onehot tensor on the right device."""
        t = torch.zeros(1, self.num_targets, dtype=torch.float32, device=self.device)
        t[0, int(target_idx)] = 1.0
        return t

    @torch.no_grad()
    def analyze_scene(self, frame_bgr: np.ndarray) -> dict:
        """
        Per-side scene recognition for the "thinking process" UI panel.
        For each half independently: softmax over all known target prompts —
        i.e. "if I had to name what's in this half, which is it?".
        Also returns per-target left-vs-right binary probabilities (the
        signal that actually drives the action).
        """
        self.policy.eval()
        x = self.preprocess_frame(frame_bgr)
        W = x.shape[3]
        mid = W // 2
        l_feat = self.policy._encode_half(x[:, :, :, :mid])
        r_feat = self.policy._encode_half(x[:, :, :, mid:])

        text = self.policy.text_features
        scale = self.policy.logit_scale.exp()

        l_sim = (l_feat @ text.T) * scale
        r_sim = (r_feat @ text.T) * scale

        l_probs = F.softmax(l_sim, dim=1).cpu().numpy()[0]
        r_probs = F.softmax(r_sim, dim=1).cpu().numpy()[0]

        binary = {}
        for t in range(self.num_targets):
            pair = torch.stack([l_sim[0, t], r_sim[0, t]], dim=0)
            binary[t] = F.softmax(pair, dim=0).cpu().numpy()

        return {
            "left_probs": l_probs,
            "right_probs": r_probs,
            "binary_probs": binary,
        }

    @torch.no_grad()
    def get_probs(self, frame_bgr: np.ndarray, target_idx: int = 0) -> np.ndarray:
        """
        Returns softmax probabilities [p_left, p_right] as numpy, conditioned on target.
        """
        self.policy.eval()
        x = self.preprocess_frame(frame_bgr)
        target = self._target_tensor(target_idx)
        logits = self.policy(x, target)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    @torch.no_grad()
    def decide(self, frame_bgr: np.ndarray, target_idx: int = 0, deterministic: bool = False) -> Tuple[int, float, np.ndarray]:
        """
        Epsilon-greedy action selection conditioned on target.
        Returns: (action_idx, log_prob_value, probs_array)
        action_idx: 0=left, 1=right
        """
        self.policy.eval()
        probs = self.get_probs(frame_bgr, target_idx)
        eps = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start
            - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * min(self.episode_count / self.cfg.epsilon_decay_steps, 1.0),
        )

        if not deterministic and np.random.rand() < eps:
            action = np.random.randint(0, 2)
            log_prob = np.log(0.5)  # dummy uniform
            return action, float(log_prob), probs

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = np.random.choice(2, p=probs)
        log_prob_val = float(np.log(probs[action] + 1e-8))
        return action, log_prob_val, probs

    def update(self, frame_bgr: np.ndarray, target_idx: int, action: int, log_prob_val: float, reward: float):
        """
        Inference-only mode: record feedback for stats/UI, no gradient step.
        CLIP backbone is frozen and prompts now match the real objects, so
        the only trainable parameter (logit_scale) was just sharpening
        confidence — letting an owner mis-press flip its sign hurts more
        than it helps. Keep the stats so the accuracy curve still works.
        """
        advantage = reward - self.baseline
        self.baseline = self.cfg.baseline_decay * self.baseline + (1 - self.cfg.baseline_decay) * reward

        self.episode_count += 1
        self.total_reward += reward
        self.reward_history.append(reward)
        self.accuracy_window.append(1 if reward > 0.5 else 0)
        self.loss_history.append(0.0)
        self.prob_history.append(float(np.exp(log_prob_val)))

        acc = np.mean(self.accuracy_window) if self.accuracy_window else 0.0
        logger.info(
            "Episode %d | reward=%.2f | baseline=%.3f | advantage=%.3f | acc=%.2f%% (inference-only)",
            self.episode_count, reward, self.baseline, advantage, acc * 100
        )

    @property
    def current_accuracy(self) -> float:
        if not self.accuracy_window:
            return 0.0
        return float(np.mean(self.accuracy_window))

    @property
    def current_eps(self) -> float:
        return max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start
            - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * min(self.episode_count / self.cfg.epsilon_decay_steps, 1.0),
        )

    def save(self, path: str):
        """Save model + training stats."""
        state = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "baseline": self.baseline,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "config": asdict(self.cfg),
            "correct_count": self.correct_count,
        }
        torch.save(state, path)
        logger.info("Model saved to %s (episodes=%d)", path, self.episode_count)

    def load(self, path: str) -> bool:
        if not torch.cuda.is_available() and self.device == "cuda":
            map_loc = "cpu"
        else:
            map_loc = self.device
        try:
            state = torch.load(path, map_location=map_loc)
            # strict=False: allow missing CLIP backbone keys (loaded fresh from open_clip in __init__)
            missing, unexpected = self.policy.load_state_dict(state["policy_state_dict"], strict=False)
            non_clip_missing = [k for k in missing if not k.startswith("clip_model.")]
            if non_clip_missing:
                logger.warning("Missing keys in checkpoint (non-CLIP): %s", non_clip_missing)
            if unexpected:
                logger.warning("Unexpected keys in checkpoint: %s", unexpected)
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as oe:
                logger.warning("Optimizer state mismatch (will use fresh optimizer): %s", oe)
            self.baseline = state.get("baseline", 0.0)
            self.episode_count = state.get("episode_count", 0)
            self.total_reward = state.get("total_reward", 0.0)
            self.correct_count = state.get("correct_count", 0)
            logger.info("Model loaded from %s (episodes=%d)", path, self.episode_count)
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False
