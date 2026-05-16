"""
Voice input via Alibaba FunASR (Paraformer-zh).
Push-to-talk: caller hands us a recorded audio file path, we return the
transcription + a routed (kind, payload) tuple ready to drive the game.

kind ∈ {"target", "feedback", "unknown"}:
  - target   → payload is target_idx (0=萝卜, 1=纸巾, 2=可乐)
  - feedback → payload is bool (True=correct/真棒, False=wrong/错了)
  - unknown  → payload is the raw transcript so the UI can show "didn't catch"
"""
from __future__ import annotations
import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Substring keyword tables. Mandarin output from FunASR is unsegmented,
# so we just check substring membership.
TARGET_KEYWORDS = [
    ("萝卜", 0), ("胡萝卜", 0), ("carrot", 0),
    ("纸巾", 1), ("卷纸", 1), ("厕纸", 1), ("卫生纸", 1), ("tissue", 1), ("paper", 1),
    ("可乐", 2), ("可口可乐", 2), ("cola", 2), ("coca", 2), ("coke", 2),
]

PRAISE_KEYWORDS = ["真棒", "对了", "答对了", "答对", "正确", "厉害", "不错", "棒棒", "对啦", "太棒"]
BLAME_KEYWORDS = ["错了", "不对", "错啦", "错的", "不是", "答错"]


class VoiceInput:
    """Wraps FunASR Paraformer-zh. Loads model lazily on first use."""

    # Default to the small Paraformer-zh — Chinese-native, ~200MB, much
    # better than Whisper-base/small at short Mandarin keywords.
    DEFAULT_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from funasr import AutoModel
            logger.info("Loading FunASR Paraformer-zh: %s (first run downloads ~200MB)", self.model_name)
            # disable_update silences ModelScope's nag about updates;
            # disable_log keeps stdout clean for our UI logs.
            self._model = AutoModel(
                model=self.model_name,
                disable_update=True,
                disable_log=True,
                disable_pbar=True,
            )
            logger.info("FunASR ready.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file → cleaned text string."""
        self._ensure_loaded()
        try:
            result = self._model.generate(
                input=audio_path,
                cache={},
                batch_size_s=60,
            )
        except Exception as e:
            logger.exception("FunASR generate failed")
            return ""
        if not result:
            return ""
        text = (result[0].get("text") or "").strip()
        # Paraformer often emits spaces between Chinese characters — strip them
        text = text.replace(" ", "")
        for ch in "。，！？.,!? ":
            text = text.rstrip(ch)
        return text


def route(text: str, awaiting_feedback: bool) -> Tuple[str, object]:
    """
    Decide what action a transcript implies, given current game context.
    Returns (kind, payload). See module docstring.
    """
    if not text:
        return ("unknown", "")
    low = text.lower()

    if awaiting_feedback:
        for kw in BLAME_KEYWORDS:
            if kw in text:
                return ("feedback", False)
        for kw in PRAISE_KEYWORDS:
            if kw in text:
                return ("feedback", True)

    for kw, idx in TARGET_KEYWORDS:
        if kw.lower() in low:
            return ("target", idx)

    if not awaiting_feedback:
        for kw in BLAME_KEYWORDS:
            if kw in text:
                return ("feedback", False)
        for kw in PRAISE_KEYWORDS:
            if kw in text:
                return ("feedback", True)

    return ("unknown", text)
