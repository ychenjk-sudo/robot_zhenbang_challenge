"""
Vision Module (Simplified for RL)
Only handles camera capture and raw frame preprocessing.
No hard-coded object detection — the neural network sees raw pixels.
"""
import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VisionSystem:
    """Handles camera capture and basic frame operations."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        logger.info("Camera started: %dx%d", self.width, self.height)

    def stop(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    @staticmethod
    def annotate_probs(frame: np.ndarray, probs: np.ndarray, choice: Optional[str] = None):
        """
        Draw probability bars on the frame for visualization.
        probs: [p_left, p_right]
        """
        out = frame.copy()
        h, w = out.shape[:2]
        bar_h = 20
        margin = 10
        max_bar_w = w // 3

        labels = ["LEFT (左)", "RIGHT (右)"]
        colors = [(0, 140, 255), (255, 0, 255)]  # orange-ish, purple-ish

        for i, (label, color) in enumerate(zip(labels, colors)):
            y = margin + i * (bar_h + margin + 20)
            # Label
            cv2.putText(out, label, (margin, y + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Bar background
            cv2.rectangle(out, (margin, y + bar_h), (margin + max_bar_w, y + bar_h + bar_h), (60, 60, 60), -1)
            # Bar fill
            fill_w = int(probs[i] * max_bar_w)
            cv2.rectangle(out, (margin, y + bar_h), (margin + fill_w, y + bar_h + bar_h), color, -1)
            # Percentage text
            cv2.putText(out, f"{probs[i]*100:.1f}%", (margin + max_bar_w + 5, y + bar_h + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if choice:
            text = f"DECIDED: {choice}"
            cv2.putText(out, text, (w // 2 - 80, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Status line
        cv2.putText(out, "ROBOT LEARNING FROM REWARD", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        return out
