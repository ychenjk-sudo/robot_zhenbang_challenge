"""
Object Detector — multi-backend perception for the Zhenbang robot game.

Backends (set via backend= parameter or ZHENBANG_DETECTOR env var):
  "vlm"    — Open-vocab via VLM, OpenAI-compatible API. Auto-picks provider:
              GLM (free)    : ZHIPU_API_KEY      → glm-4v-flash
              Qwen (cheap)  : DASHSCOPE_API_KEY  → qwen-vl-plus
              OpenAI        : OPENAI_API_KEY     → gpt-4o-mini
              Anthropic     : ANTHROPIC_API_KEY  → claude-haiku-4-5
              Ollama (local): OLLAMA_BASE_URL    → moondream
  "color"  — HSV color segmentation. Zero deps, real-time, needs tuning.
  "yolo"   — YOLOv8 COCO (pip install ultralytics).
  "owlvit" — OWL-ViT, fixed threshold + CPU.
  "auto"   — vlm → yolo → color in order.

Same detect() / best_for_target() / annotate() interface throughout.
"""
from __future__ import annotations
import logging
import os
import threading
from typing import List, Sequence, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Object definitions
# ---------------------------------------------------------------------------
# Chinese names, English descriptions for VLM prompt, HSV ranges for fallback
_OBJECTS = [
    {
        "zh": "萝卜",
        "en": "carrot",
        "desc": "an orange carrot vegetable",
        # HSV: H 5-22, S>120, V>80
        "hsv": [(np.array([5, 120, 80]), np.array([22, 255, 255]))],
    },
    {
        "zh": "纸巾",
        "en": "tissue roll",
        "desc": "a white toilet paper roll or tissue roll",
        # HSV: any H, very low saturation, high value (white)
        "hsv": [(np.array([0, 0, 190]), np.array([180, 40, 255]))],
    },
    {
        "zh": "可乐",
        "en": "cola can",
        "desc": "a red Coca-Cola can",
        # HSV: red wraps around H=0
        "hsv": [
            (np.array([0,  100, 60]), np.array([10, 255, 220])),
            (np.array([165, 100, 60]), np.array([180, 255, 220])),
        ],
    },
]
_MIN_BLOB_AREA = 600   # pixels² for color backend

# COCO class → object index (yolo backend)
_YOLO_CLASS_MAP: Dict[int, int] = {56: 0, 39: 2, 44: 2}  # carrot, bottle, cup


# ---------------------------------------------------------------------------
# VLM backend  (OpenAI-compatible: GLM / Qwen / OpenAI / Claude / Ollama)
# ---------------------------------------------------------------------------

# Provider config — only Qwen2.5-VL (true bbox grounding).
# Override model with ZHENBANG_VLM_MODEL=qwen2.5-vl-7b-instruct (cheaper but
# slightly less reliable on JSON structure) or qwen2.5-vl-72b-instruct (default).
_VLM_PROVIDERS = [
    # 7B is 4× faster & cheaper than 72B; with our few-shot prompt +
    # tolerant parser, JSON reliability is fine. Switch to 72B via
    # ZHENBANG_VLM_MODEL=qwen2.5-vl-72b-instruct only if you see drift.
    ("qwen", "DASHSCOPE_API_KEY",
     "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "qwen2.5-vl-7b-instruct"),
]


def _pick_vlm_provider():
    """Return (provider_name, api_key, base_url, model). Raises if none configured."""
    forced = os.environ.get("ZHENBANG_VLM_PROVIDER")
    for name, env_var, base_url, model in _VLM_PROVIDERS:
        if forced and forced != name:
            continue
        val = os.environ.get(env_var)
        if val:
            # ollama: env var IS the base url, key is dummy
            if name == "ollama":
                return name, "ollama", val.rstrip("/") + "/v1", os.environ.get("ZHENBANG_VLM_MODEL", model)
            return name, val, base_url, os.environ.get("ZHENBANG_VLM_MODEL", model)
    raise RuntimeError(
        "VLM not configured. Set DASHSCOPE_API_KEY for Qwen2.5-VL "
        "(get key at https://bailian.console.aliyun.com)."
    )


# Qwen2.5-VL bbox grounding prompt — few-shot example helps small (7B) models
# stick to the schema. Realistic example mirrors the carrot/cola/tissue scene.
_VLM_PROMPT_BBOX = """\
检测图中物体并输出精确像素 bbox。

物体类别：
  0 = 萝卜（橙色胡萝卜，真实蔬菜）
  1 = 纸巾（白色卷纸/卫生纸卷）
  2 = 可乐（红色可口可乐易拉罐）

图片尺寸 {W}x{H} 像素。

输出格式：严格 JSON，不要 markdown，不要解释。

示例输入：图中桌上从左到右是萝卜、可乐、纸巾。
示例输出：
{{"objects":[{{"idx":0,"bbox":[57,263,224,429]}},{{"idx":2,"bbox":[278,139,430,397]}},{{"idx":1,"bbox":[473,108,811,455]}}]}}

规则：
- bbox 必须是 4 个整数 [x1, y1, x2, y2]，绝对像素坐标
- 只输出真实存在的物体
- 三个物体如果都在，objects 数组要有 3 项"""


def _strip_to_json(raw: str) -> str:
    """Pull JSON out of code fences / prose, accept either {...} or [...]."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:]).rstrip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    raw = raw.strip()
    # Find outermost JSON container (object or array)
    if raw and raw[0] not in "{[":
        s_obj, s_arr = raw.find("{"), raw.find("[")
        starts = [p for p in (s_obj, s_arr) if p != -1]
        if not starts:
            return raw
        s = min(starts)
        e_obj, e_arr = raw.rfind("}"), raw.rfind("]")
        e = max(e_obj, e_arr)
        if e > s:
            raw = raw[s:e+1]
    return raw


def _repair_json(raw: str) -> str:
    """Best-effort fixes for common small-model glitches."""
    import re
    # 1. Stray quote before }: e.g. [1,2,3,4]"} → [1,2,3,4]}
    raw = re.sub(r'(\])\s*"\s*\}', r'\1}', raw)
    # 2. Trailing commas: ,] or ,}
    raw = re.sub(r',(\s*[\]}])', r'\1', raw)
    # 3. Single quotes around keys (rare)
    raw = re.sub(r"'(\w+)'(\s*):", r'"\1"\2:', raw)
    return raw


def _parse_qwen_objects(raw_text: str) -> list:
    """Parse Qwen bbox output. Tolerant of: bare arrays, repair fixes,
    objects vs detections key, bbox vs bbox_2d field."""
    import json
    raw = _strip_to_json(raw_text)
    for attempt in (raw, _repair_json(raw)):
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            data = None
    if data is None:
        return []
    # Accept both {"objects":[...]} and bare [...]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("objects") or data.get("detections") or []
    else:
        return []
    out = []
    for o in items:
        if not isinstance(o, dict):
            continue
        bbox = o.get("bbox") or o.get("bbox_2d") or o.get("box")
        idx  = o.get("idx", o.get("class_id", o.get("label_id")))
        if bbox is None or idx is None:
            continue
        try:
            idx = int(idx)
            if len(bbox) != 4:
                continue
            out.append({"idx": idx,
                        "bbox": [int(x) for x in bbox],
                        "score": float(o.get("score", 0.9))})
        except (ValueError, TypeError):
            continue
    return out


# Dynamic single-target prompt — for "find ANY object the user types in"
_VLM_PROMPT_DYNAMIC = """\
请在图片中找到 "{target}"。

图片尺寸 {W}x{H} 像素。

输出格式：严格 JSON，不要 markdown，不要解释。

如果找到：
{{"found": true, "bbox": [x1, y1, x2, y2], "confidence": 0.95}}

如果没找到：
{{"found": false}}

bbox 是绝对像素坐标 [左上x, 左上y, 右下x, 右下y]。
只返回最显眼的那一个目标。"""


def _vlm_detect_target(frame_bgr: np.ndarray, target_text: str,
                       provider_info) -> Optional[Dict]:
    """Find one object by free-text query. Returns single dict or None."""
    import base64, json, cv2
    from openai import OpenAI

    provider, api_key, base_url, model = provider_info
    H, W = frame_bgr.shape[:2]
    target_w = 800
    sent_W, sent_H = W, H
    send_img = frame_bgr
    if W > target_w:
        scale = target_w / W
        sent_W, sent_H = target_w, int(H * scale)
        send_img = cv2.resize(frame_bgr, (sent_W, sent_H))
    _, buf = cv2.imencode(".jpg", send_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()

    prompt = _VLM_PROMPT_DYNAMIC.format(target=target_text, W=sent_W, H=sent_H)
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    kwargs = dict(
        model=model, max_tokens=300, temperature=0.0,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    try:
        response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except Exception:
        response = client.chat.completions.create(**kwargs)

    raw = _strip_to_json(response.choices[0].message.content)
    raw = _repair_json(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not data.get("found"):
        return None
    bbox = data.get("bbox") or data.get("bbox_2d")
    if not bbox or len(bbox) != 4:
        return None

    sx, sy = W / sent_W, H / sent_H
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W, int(x1 * sx)))
    y1 = max(0, min(H, int(y1 * sy)))
    x2 = max(0, min(W, int(x2 * sx)))
    y2 = max(0, min(H, int(y2 * sy)))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "name_idx": -1,            # dynamic targets don't have a fixed idx
        "name": target_text,
        "bbox": (x1, y1, x2, y2),
        "score": float(data.get("confidence", 0.9)),
        "cx": (x1 + x2) // 2,
    }


def _vlm_detect(frame_bgr: np.ndarray, names: List[str], provider_info) -> List[Dict]:
    import base64, cv2
    from openai import OpenAI

    provider, api_key, base_url, model = provider_info
    H, W = frame_bgr.shape[:2]

    # Qwen2.5-VL grounding does better at 800px+; smaller is too lossy for bbox.
    target_w = 800
    sent_W, sent_H = W, H
    send_img = frame_bgr
    if W > target_w:
        scale = target_w / W
        sent_W, sent_H = target_w, int(H * scale)
        send_img = cv2.resize(frame_bgr, (sent_W, sent_H))
    _, buf = cv2.imencode(".jpg", send_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    kwargs = dict(
        model=model,
        max_tokens=600,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _VLM_PROMPT_BBOX.format(W=sent_W, H=sent_H)},
            ],
        }],
    )
    # DashScope supports response_format json_object for newer models — try it,
    # fall back without if rejected.
    try:
        response = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"})
    except Exception:
        response = client.chat.completions.create(**kwargs)

    items = _parse_qwen_objects(response.choices[0].message.content)

    sx = W / sent_W
    sy = H / sent_H
    detections: List[Dict] = []
    for o in items:
        idx = o["idx"]
        if idx < 0 or idx >= len(names):
            continue
        x1, y1, x2, y2 = o["bbox"]
        x1 = max(0, min(W, int(x1 * sx)))
        y1 = max(0, min(H, int(y1 * sy)))
        x2 = max(0, min(W, int(x2 * sx)))
        y2 = max(0, min(H, int(y2 * sy)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "name_idx": idx,
            "name": names[idx],
            "bbox": (x1, y1, x2, y2),
            "score": o["score"],
            "cx": (x1 + x2) // 2,
        })
    detections.sort(key=lambda d: -d["score"])
    return detections


# ---------------------------------------------------------------------------
# Color backend
# ---------------------------------------------------------------------------

def _color_detect_one(frame_bgr: np.ndarray, idx: int) -> Optional[Dict]:
    import cv2
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _OBJECTS[idx]["hsv"]:
        mask |= cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < _MIN_BLOB_AREA:
        return None
    x, y, w, h = cv2.boundingRect(best)
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(W, x + w); y2 = min(H, y + h)
    score = min(1.0, cv2.contourArea(best) / (W * H * 0.25))
    return {"name_idx": idx, "bbox": (x1, y1, x2, y2), "score": float(score)}


# ---------------------------------------------------------------------------
# YOLO backend
# ---------------------------------------------------------------------------

def _yolo_detect_all(model, frame_bgr: np.ndarray, names: List[str]) -> List[Dict]:
    results = model(frame_bgr, verbose=False)[0]
    H, W = frame_bgr.shape[:2]
    dets: List[Dict] = []
    for box in results.boxes:
        game_idx = _YOLO_CLASS_MAP.get(int(box.cls.item()))
        if game_idx is None:
            continue
        score = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
        dets.append({"name_idx": game_idx, "name": names[game_idx],
                     "bbox": (x1, y1, x2, y2), "score": score})
    dets.sort(key=lambda d: -d["score"])
    return dets


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class ObjectDetector:
    DEFAULT_MODEL = "google/owlvit-base-patch32"

    def __init__(
        self,
        prompts: Sequence[str],
        names: Optional[Sequence[str]] = None,
        model_name: str = DEFAULT_MODEL,
        score_threshold: float = 0.005,
        device: Optional[str] = None,
        backend: str = "auto",
    ):
        self.prompts: List[str] = list(prompts)
        self.names: List[str] = list(names) if names else list(prompts)
        if len(self.names) != len(self.prompts):
            raise ValueError("names and prompts must have same length")
        self.model_name   = model_name
        self.score_threshold = score_threshold
        self.backend      = os.environ.get("ZHENBANG_DETECTOR", backend)

        import torch
        self.device = device or ("cpu" if self.backend == "owlvit"
                                 else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self._model = None
        self._processor = None
        self._load_lock  = threading.Lock()
        self._active_backend: Optional[str] = None

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            order = self._backend_order()
            for b in order:
                try:
                    self._try_load(b)
                    return
                except Exception as e:
                    logger.warning("Backend %s failed to load: %s", b, e)
            raise RuntimeError("All detector backends failed to load.")

    def _backend_order(self) -> List[str]:
        if self.backend == "auto":
            order = []
            if any(os.environ.get(v) for v in (
                "ZHIPU_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL")):
                order.append("vlm")
            order += ["yolo", "color"]
            return order
        return [self.backend]

    def _try_load(self, b: str):
        if b == "vlm":
            import openai  # noqa — just test import
            self._vlm_provider = _pick_vlm_provider()
            self._model = "vlm"
            self._active_backend = "vlm"
            logger.info("VLM backend ready (provider=%s, model=%s).",
                        self._vlm_provider[0], self._vlm_provider[3])
        elif b == "yolo":
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")
            self._active_backend = "yolo"
            logger.info("YOLO backend ready.")
        elif b == "color":
            self._model = "color"
            self._active_backend = "color"
            logger.info("Color-segmentation backend ready.")
        elif b == "owlvit":
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            logger.info("Loading OWL-ViT on %s ...", self.device)
            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(
                self.model_name).to(self.device)
            self._model.eval()
            self._active_backend = "owlvit"
            logger.info("OWL-ViT ready on %s.", self.device)
        else:
            raise ValueError(f"Unknown backend: {b}")

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, top_k_per_class: int = 1) -> List[Dict]:
        self._ensure_loaded()
        if self._active_backend == "vlm":
            dets = _vlm_detect(frame_bgr, self.names, self._vlm_provider)
        elif self._active_backend == "yolo":
            dets = self._detect_yolo(frame_bgr)
        elif self._active_backend == "color":
            dets = self._detect_color(frame_bgr)
        elif self._active_backend == "owlvit":
            dets = self._detect_owlvit(frame_bgr)
        else:
            return []
        return self._topk(dets, top_k_per_class)

    def _detect_color(self, frame_bgr):
        dets = []
        for idx in range(len(self.names)):
            d = _color_detect_one(frame_bgr, idx)
            if d:
                d["name"] = self.names[idx]
                dets.append(d)
        dets.sort(key=lambda d: -d["score"])
        return dets

    def _detect_yolo(self, frame_bgr):
        dets = _yolo_detect_all(self._model, frame_bgr, self.names)
        # tissue (idx=1) has no COCO class — color fallback
        if not any(d["name_idx"] == 1 for d in dets):
            td = _color_detect_one(frame_bgr, 1)
            if td:
                td["name"] = self.names[1]
                dets.append(td)
                dets.sort(key=lambda d: -d["score"])
        return dets

    def _detect_owlvit(self, frame_bgr):
        import torch, cv2
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(
            text=[self.prompts], images=rgb, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([[H, W]], device=self.device),
            threshold=self.score_threshold,
        )[0]
        dets = []
        for s, l, b in zip(results["scores"].tolist(),
                            results["labels"].tolist(),
                            results["boxes"].tolist()):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
            dets.append({"name_idx": int(l), "name": self.names[int(l)],
                         "bbox": (x1, y1, x2, y2), "score": float(s)})
        dets.sort(key=lambda d: -d["score"])
        return dets

    @staticmethod
    def _topk(dets: List[Dict], k: int) -> List[Dict]:
        kept, count = [], {}
        for d in dets:
            i = d["name_idx"]
            if count.get(i, 0) < k:
                kept.append(d)
                count[i] = count.get(i, 0) + 1
        return kept

    # ------------------------------------------------------------------
    def detect_target(self, frame_bgr: np.ndarray, target_text: str) -> Optional[Dict]:
        """
        Open-vocab single-target search. Use this for arbitrary user-typed
        object names. Requires VLM backend.
        """
        self._ensure_loaded()
        if self._active_backend != "vlm":
            logger.warning("detect_target requires VLM backend, current=%s",
                           self._active_backend)
            return None
        return _vlm_detect_target(frame_bgr, target_text, self._vlm_provider)

    # ------------------------------------------------------------------
    def best_for_target(self, detections: List[Dict], target_idx: int) -> Optional[Dict]:
        for d in detections:
            if d["name_idx"] == target_idx:
                return d
        return None

    @staticmethod
    def annotate(frame_bgr: np.ndarray, detections: List[Dict],
                 target_idx: Optional[int] = None) -> np.ndarray:
        import cv2
        out = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            is_target = (target_idx is not None and d["name_idx"] == target_idx)
            color = (0, 255, 0) if is_target else (255, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if is_target else 2)
            label = f"{d['name']} {d['score']*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1-th-6)), (x1+tw+4, y1), color, -1)
            cv2.putText(out, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out
