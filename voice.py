"""
Voice Module
Plays Minion-style emotional sound effects from sounds/ directory.
File naming convention: <emotion>_<NN>.mp3 — e.g. excited_01.mp3, excited_02.mp3.
Multiple variants per emotion are picked at random for variety.
"""
import os
import random
import logging
from collections import defaultdict
from typing import Optional, Dict, List
from dataclasses import dataclass

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SoundConfig:
    sounds_dir: str = "./sounds"
    volume: float = 0.8
    fallback_tts: bool = False  # kept for backward-compat with config.yaml; unused


class VoiceSystem:
    """
    Loads all <emotion>_*.mp3 files from sounds_dir and plays a random
    variant when play_emotion(name) is called.
    """

    def __init__(self, config: Optional[SoundConfig] = None):
        self.cfg = config or SoundConfig()
        self.banks: Dict[str, List["pygame.mixer.Sound"]] = defaultdict(list)
        self._initialized = False

    def start(self):
        if not PYGAME_AVAILABLE:
            logger.warning("pygame not installed; audio playback disabled.")
            return
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self._load_sounds()
        self._initialized = True

    def stop(self):
        if self._initialized and PYGAME_AVAILABLE:
            pygame.mixer.quit()

    def _load_sounds(self):
        """Scan sounds_dir, group by prefix before '_NN.mp3'."""
        if not os.path.isdir(self.cfg.sounds_dir):
            logger.warning("Sounds dir not found: %s", self.cfg.sounds_dir)
            return
        for fname in sorted(os.listdir(self.cfg.sounds_dir)):
            if not fname.lower().endswith(".mp3"):
                continue
            stem = os.path.splitext(fname)[0]
            # Strip a trailing _NN suffix if present (e.g. "excited_02" -> "excited")
            parts = stem.rsplit("_", 1)
            emotion = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
            path = os.path.join(self.cfg.sounds_dir, fname)
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(self.cfg.volume)
                self.banks[emotion].append(snd)
                logger.info("Loaded %s -> emotion '%s'", fname, emotion)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

        if not self.banks:
            logger.warning("No sounds loaded from %s", self.cfg.sounds_dir)
        else:
            summary = ", ".join(f"{k}×{len(v)}" for k, v in self.banks.items())
            logger.info("Voice banks ready: %s", summary)

    def play_emotion(self, emotion: str, block: bool = False):
        """Play a random variant from the named emotion bank."""
        if not self._initialized:
            return
        bank = self.banks.get(emotion)
        if not bank:
            logger.debug("No sounds for emotion '%s'", emotion)
            return
        snd = random.choice(bank)
        chan = snd.play()
        if block and chan:
            while chan.get_busy():
                pygame.time.wait(50)
