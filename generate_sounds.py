"""
Generate Minion-style emotional sound effects via ElevenLabs Sound Generation API.

Usage:
    export ELEVENLABS_API_KEY=sk_xxx
    python generate_sounds.py

Idempotent: skips files that already exist.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://api.elevenlabs.io/v1/sound-generation"
SOUNDS_DIR = Path(__file__).parent / "sounds"
VARIANTS_PER_EMOTION = 3

# Each emotion: (prompt, duration_sec)
# Prompts are written to lean toward minion-style cartoon gibberish.
EMOTIONS = {
    "think": (
        "A small yellow cartoon mascot humming and pondering, gentle 'hmmm... bee-do bee-do' babble, "
        "curious thinking sound, no real words, playful tone",
        1.6,
    ),
    "observe": (
        "A small yellow cartoon character peering and inspecting something, short curious 'oooh? eh?' babble, "
        "inquisitive peeking sound, no real words",
        0.9,
    ),
    "hesitate": (
        "A nervous cartoon character making a hesitant 'uh oh... eh-eh...' babble, "
        "uncertain wobble in pitch, no real words, comedic timing",
        1.2,
    ),
    "act": (
        "An excited cartoon character shouting a quick 'BAH! ba-na-na!' battle cry, "
        "short punchy gibberish, no real words, playful exclamation",
        0.9,
    ),
    "excited": (
        "A cheerful cartoon mascot celebrating with 'WOOHOO! tulaliloo! yay!' happy babble, "
        "rising pitch laughter, no real words, joyful gibberish",
        1.8,
    ),
    "sad": (
        "A sad cartoon character making a disappointed 'awwww... bee-doo...' descending whimper, "
        "downcast murmur, no real words, comedic frown",
        1.6,
    ),
    "confused": (
        "A puzzled cartoon character looking around in confusion, 'huh? where? eh?' rising-tone babble, "
        "looking-for-something sound, no real words, comedic shrug feel",
        1.4,
    ),
}


def generate_one(api_key: str, prompt: str, duration: float, out_path: Path) -> None:
    payload = {
        "text": prompt,
        "duration_seconds": duration,
        "prompt_influence": 0.5,
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:300]}")
    out_path.write_bytes(resp.content)


def main() -> int:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        print("ERROR: set ELEVENLABS_API_KEY environment variable.", file=sys.stderr)
        return 1

    SOUNDS_DIR.mkdir(exist_ok=True)
    total = len(EMOTIONS) * VARIANTS_PER_EMOTION
    done = 0
    skipped = 0
    failed = 0

    for emotion, (prompt, duration) in EMOTIONS.items():
        for i in range(1, VARIANTS_PER_EMOTION + 1):
            out = SOUNDS_DIR / f"{emotion}_{i:02d}.mp3"
            if out.exists() and out.stat().st_size > 1000:
                print(f"[skip] {out.name} already exists ({out.stat().st_size} bytes)")
                skipped += 1
                continue
            print(f"[gen ] {out.name}  (duration={duration}s) ...", flush=True)
            try:
                generate_one(key, prompt, duration, out)
                size_kb = out.stat().st_size / 1024
                print(f"       → {size_kb:.1f} KB ✓")
                done += 1
                time.sleep(0.5)  # be nice to the API
            except Exception as e:
                print(f"       FAILED: {e}", file=sys.stderr)
                failed += 1

    print(f"\nDone. generated={done}, skipped={skipped}, failed={failed}, total={total}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
