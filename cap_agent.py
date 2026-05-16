"""
Phase I — Code-as-Policies Agent (Layer 3 of CaP architecture).

Architecture:
  Layer 3 (this file)   — Plan + Reflect via Qwen LLM (or hardcoded fallback)
  Layer 2 (primitives.py) — Closed-loop primitives (approach, grasp_with_retry, ...)
  Layer 1 (robot_actor)  — lerobot smooth motion + Feetech driver

The agent does NOT execute arbitrary Python code. Instead it picks primitive
names + JSON kwargs from a whitelist — safe by construction. The LLM only
chooses which primitive to call with what arguments.

LLM reuse: same provider as detector.py (DASHSCOPE_API_KEY → Qwen). We use
Qwen-Plus (text-only, fast, cheap) for plan/reflect, NOT Qwen-VL.

Design points:
  - Hardcoded plans for known tasks (萝卜/纸巾/可乐) — works WITHOUT LLM
  - LLM plan only kicks in for free-text / unknown tasks
  - Reflect uses hardcoded rules for common failure modes (cheap),
    falls through to LLM only for novel failures
  - Memory tracks last 20 attempts so reflect can see history
"""
from __future__ import annotations
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — drop locations in normalized world coordinates [0,1]²
# These match the calibrated workspace. Tune if your calibration is different.
# ---------------------------------------------------------------------------

LEFT_DROP   = (0.15, 0.55)   # left side of table, mid depth
CENTER_DROP = (0.50, 0.55)
RIGHT_DROP  = (0.85, 0.55)

DROP_LOCATIONS = {
    "left":   LEFT_DROP,
    "center": CENTER_DROP,
    "right":  RIGHT_DROP,
    "左":     LEFT_DROP,
    "中":     CENTER_DROP,
    "右":     RIGHT_DROP,
}


# ---------------------------------------------------------------------------
# Hardcoded plans — used as default and as LLM fallback
# Each item: (primitive_name, kwargs_dict)
# ---------------------------------------------------------------------------

def _carrot_pick_left_plan() -> List[Tuple[str, dict]]:
    return [
        ("approach",          {"target_name": "carrot", "max_iters": 3}),
        ("grasp_with_retry",  {"target_name": "carrot", "max_attempts": 3}),
        ("transport_to",      {"dest_world_xy": LEFT_DROP}),
        ("release",           {}),
        ("go_home",           {}),
    ]


def _generic_pick_plan(target_name: str,
                       drop: Tuple[float, float] = LEFT_DROP
                       ) -> List[Tuple[str, dict]]:
    return [
        ("approach",          {"target_name": target_name, "max_iters": 3}),
        ("grasp_with_retry",  {"target_name": target_name, "max_attempts": 3}),
        ("transport_to",      {"dest_world_xy": drop}),
        ("release",           {}),
        ("go_home",           {}),
    ]


HARDCODED_PLANS = {
    # Chinese keys
    "萝卜": _carrot_pick_left_plan,
    "carrot": _carrot_pick_left_plan,
    "拿萝卜": _carrot_pick_left_plan,
    "拿萝卜放左边": _carrot_pick_left_plan,
}


# ---------------------------------------------------------------------------
# Whitelist of primitives the LLM is allowed to call.
# Every primitive is documented for the LLM prompt.
# ---------------------------------------------------------------------------

PRIMITIVE_REGISTRY = {
    "approach": {
        "doc": "approach(target_name: str, max_iters: int = 3) — 闭环靠近目标. "
               "回到 observe pose → 检测 → 移到目标上方 hover → 重复. 收敛或达到 max_iters 时返回.",
        "required_args": ["target_name"],
        "arg_types": {"target_name": str, "max_iters": int},
    },
    "grasp_with_retry": {
        "doc": "grasp_with_retry(target_name: str = 'carrot', max_attempts: int = 3) — "
               "下降到 hit pose → 闭夹爪 → 用夹爪角度验证 → 失败则微抖动+下降更深 重试.",
        "required_args": [],
        "arg_types": {"target_name": str, "max_attempts": int},
    },
    "transport_to": {
        "doc": "transport_to(dest_world_xy: [float, float]) — "
               "(假设已抓住物体) 移到 (x, y) ∈ [0,1]² 在 hover 高度.",
        "required_args": ["dest_world_xy"],
        "arg_types": {"dest_world_xy": (list, tuple)},
    },
    "release": {
        "doc": "release() — 张开夹爪释放物体.",
        "required_args": [],
        "arg_types": {},
    },
    "open_gripper": {
        "doc": "open_gripper() — 张开夹爪 (不释放检测物, 仅控制夹爪).",
        "required_args": [],
        "arg_types": {},
    },
    "close_gripper": {
        "doc": "close_gripper() — 闭合夹爪.",
        "required_args": [],
        "arg_types": {},
    },
    "go_home": {
        "doc": "go_home() — 回到 home pose.",
        "required_args": [],
        "arg_types": {},
    },
    "go_to_observe": {
        "doc": "go_to_observe() — 回到 observe pose (homography 标定的视角).",
        "required_args": [],
        "arg_types": {},
    },
}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@dataclass
class AgentMemory:
    """Short-term memory of recent attempts. In-process only, not persisted."""
    attempts: List[Dict] = field(default_factory=list)

    def log(self, task: str, plan_summary: List[str], outcome: str, info: str = ""):
        self.attempts.append({
            "ts": time.time(),
            "task": task,
            "plan": plan_summary,
            "outcome": outcome,   # "success" / "failed" / "cancelled"
            "info": info,
        })
        if len(self.attempts) > 20:
            self.attempts = self.attempts[-20:]

    def recent(self, n: int = 3) -> List[Dict]:
        return self.attempts[-n:]

    def recent_failures(self, n: int = 3) -> List[Dict]:
        return [a for a in self.attempts if a["outcome"] != "success"][-n:]


# ---------------------------------------------------------------------------
# PickAgent
# ---------------------------------------------------------------------------

class PickAgent:
    """
    Code-as-Policies agent. Composes primitives into multi-step tasks
    with LLM-based planning + reflection.

    Constructor args
    ----------------
    primitives : CaPPrimitives  — Layer 2 instance
    detector   : ObjectDetector — used to look up the same VLM provider info
    use_llm    : bool — if False, always use hardcoded plans (no API calls)
    state_callback : optional Callable[step_name, info] — UI live updates
    """

    def __init__(self,
                 primitives,
                 detector=None,
                 use_llm: bool = True,
                 state_callback: Optional[Callable[[str, str], None]] = None):
        self.p = primitives
        self.detector = detector
        self.use_llm = use_llm
        self.state_callback = state_callback
        self.memory = AgentMemory()

        # Cancellation flag — UI can set this from another thread
        self._cancel = False

        # Resolve LLM provider from detector (it has _vlm_provider tuple after first detect)
        self._llm_provider: Optional[Tuple[str, str, str, str]] = None
        if self.use_llm:
            self._llm_provider = self._resolve_llm_provider()

        # Telemetry (UI reads these)
        self.last_task: str = ""
        self.last_plan: List[Tuple[str, dict]] = []
        self.last_step: str = ""
        self.last_outcome: str = ""

    # ------------------------------------------------------------------
    def cancel(self):
        self._cancel = True

    def reset(self):
        self._cancel = False
        self.last_step = ""
        self.last_outcome = ""

    def _emit(self, step: str, info: str = ""):
        self.last_step = step
        if self.state_callback:
            try:
                self.state_callback(step, info)
            except Exception as e:
                logger.warning("state_callback raised: %s", e)

    # ------------------------------------------------------------------
    def _resolve_llm_provider(self):
        """
        Try to resolve a Qwen API config. Reuses detector.py's _pick_vlm_provider
        if available, otherwise reads DASHSCOPE_API_KEY directly.
        """
        # Method A: detector already has it
        if self.detector is not None:
            prov = getattr(self.detector, "_vlm_provider", None)
            if prov:
                return prov
        # Method B: ask detector module
        try:
            from detector import _pick_vlm_provider  # type: ignore
            return _pick_vlm_provider()
        except Exception as e:
            logger.info("LLM provider not available, using hardcoded plans only: %s", e)
            return None

    # ------------------------------------------------------------------
    # PUBLIC: run a task end-to-end
    # ------------------------------------------------------------------

    def run(self, task: str) -> Tuple[bool, str]:
        """
        Execute a task end-to-end. Returns (success, description).

        Steps:
          1. Plan (hardcoded if known, else LLM, else default fallback)
          2. Execute each primitive in order
          3. On primitive failure: reflect → recovery primitive → continue
          4. Log to memory
        """
        self.reset()
        self.last_task = task

        # --- Plan ---
        plan = self._make_plan(task)
        if not plan:
            self.memory.log(task, [], "failed", "no plan generated")
            return False, "无法生成 plan"

        self.last_plan = list(plan)
        plan_summary = [f"{n}({_kwargs_summary(kw)})" for n, kw in plan]
        logger.info("Agent plan for %r: %s", task, plan_summary)
        self._emit("plan", " → ".join(plan_summary))

        # --- Execute ---
        for step_i, (prim_name, prim_kwargs) in enumerate(plan):
            if self._cancel:
                self.memory.log(task, plan_summary, "cancelled")
                return False, "已取消"

            self._emit("exec", f"step {step_i+1}/{len(plan)}: {prim_name}")
            ok, info = self._exec_primitive(prim_name, prim_kwargs)
            logger.info("[step %d] %s → %s | %s", step_i+1, prim_name,
                        "OK" if ok else "FAIL", info)

            if ok:
                continue

            # --- Reflect on failure ---
            self._emit("reflect", f"failed: {prim_name} | {info}")
            recovery = self._reflect(task, plan, step_i, prim_name, info)

            if recovery is None:
                self.memory.log(task, plan_summary, "failed",
                                f"step={prim_name}: {info}")
                self.last_outcome = "failed"
                return False, f"失败于 {prim_name}: {info}"

            r_name, r_kwargs = recovery
            self._emit("recover", f"trying {r_name}")
            r_ok, r_info = self._exec_primitive(r_name, r_kwargs)
            logger.info("[recover] %s → %s | %s", r_name,
                        "OK" if r_ok else "FAIL", r_info)
            if not r_ok:
                self.memory.log(task, plan_summary, "failed",
                                f"recovery_failed step={prim_name}: {r_info}")
                self.last_outcome = "failed"
                return False, f"恢复失败于 {prim_name}→{r_name}: {r_info}"

        self.memory.log(task, plan_summary, "success")
        self.last_outcome = "success"
        self._emit("done", "task completed")
        return True, "任务完成"

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------

    def _make_plan(self, task: str) -> List[Tuple[str, dict]]:
        """Resolve a plan: hardcoded → LLM → generic fallback."""
        # 1) Hardcoded exact match
        key = task.strip()
        if key in HARDCODED_PLANS:
            return HARDCODED_PLANS[key]()
        # 1b) Hardcoded substring (for "拿萝卜" matching "拿萝卜放左边")
        for k, factory in HARDCODED_PLANS.items():
            if k in key or k.lower() in key.lower():
                return factory()

        # 2) LLM plan (only if configured)
        if self.use_llm and self._llm_provider:
            try:
                plan = self._llm_plan(task)
                if plan and self._validate_plan(plan):
                    return plan
                logger.warning("LLM returned empty/invalid plan, using fallback")
            except Exception as e:
                logger.warning("LLM plan call failed: %s — using fallback", e)

        # 3) Default fallback: treat task as a target name, drop on left
        return _generic_pick_plan(task, drop=LEFT_DROP)

    # ------------------------------------------------------------------
    SYSTEM_PROMPT = """你是一个机器人任务规划器. 给定一个用自然语言描述的任务,
输出一个 JSON 格式的 primitive 调用序列.

可用的 primitive (函数名 → 文档):
{primitive_docs}

预设位置 (世界坐标 [0,1]²):
- LEFT_DROP   = [0.15, 0.55]
- CENTER_DROP = [0.50, 0.55]
- RIGHT_DROP  = [0.85, 0.55]

输出格式 (严格 JSON, 无解释, 无 markdown 围栏):
{{"plan": [
    {{"primitive": "approach", "kwargs": {{"target_name": "carrot"}}}},
    {{"primitive": "grasp_with_retry", "kwargs": {{}}}},
    {{"primitive": "transport_to", "kwargs": {{"dest_world_xy": [0.15, 0.55]}}}},
    {{"primitive": "release", "kwargs": {{}}}},
    {{"primitive": "go_home", "kwargs": {{}}}}
]}}

规则:
- target_name 用英文 (carrot / tissue / cola / 等). 中文物体名也可以.
- dest_world_xy 必须是 [x, y] 数组, 0<=x,y<=1.
- 一定要以 go_home 结尾 (除非是 partial 任务).

最近 3 次失败案例 (用于避免重复错误):
{recent_failures}

任务: {task}"""

    def _llm_plan(self, task: str) -> List[Tuple[str, dict]]:
        """Ask Qwen-Plus (text) to generate a plan."""
        from openai import OpenAI

        provider, api_key, base_url, _vl_model = self._llm_provider
        # Use a TEXT model (faster + cheaper than VL for plan gen)
        text_model = os.environ.get("ZHENBANG_AGENT_MODEL", "qwen-plus")

        # Build prompt
        prim_docs = "\n".join(
            f"  - {info['doc']}" for info in PRIMITIVE_REGISTRY.values()
        )
        recent = self.memory.recent_failures(n=3)
        if recent:
            failures_str = "\n".join(
                f"  - task={a['task']!r}: {a['info']}" for a in recent
            )
        else:
            failures_str = "(无)"

        prompt = self.SYSTEM_PROMPT.format(
            primitive_docs=prim_docs,
            recent_failures=failures_str,
            task=task,
        )

        client = (OpenAI(api_key=api_key, base_url=base_url)
                  if base_url else OpenAI(api_key=api_key))

        kwargs = dict(
            model=text_model, max_tokens=600, temperature=0.0,
            messages=[
                {"role": "system",
                 "content": "你是一个机器人任务规划器. 严格输出 JSON, 不解释."},
                {"role": "user", "content": prompt},
            ],
        )
        try:
            resp = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"})
        except Exception:
            resp = client.chat.completions.create(**kwargs)

        raw = resp.choices[0].message.content or ""
        return self._parse_plan(raw)

    @staticmethod
    def _parse_plan(raw: str) -> List[Tuple[str, dict]]:
        """Parse LLM output → list of (primitive_name, kwargs) tuples."""
        raw = raw.strip()
        # Strip markdown fence
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rstrip("` \n")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
        # Find outermost JSON object
        s = raw.find("{")
        e = raw.rfind("}")
        if s < 0 or e <= s:
            return []
        body = raw[s:e+1]
        # Common cleanups
        body = re.sub(r",(\s*[\]}])", r"\1", body)   # trailing commas
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return []

        steps = data.get("plan") or data.get("steps") or []
        out: List[Tuple[str, dict]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            name = step.get("primitive") or step.get("name") or step.get("call")
            kwargs = step.get("kwargs") or step.get("args") or {}
            if not isinstance(name, str) or not isinstance(kwargs, dict):
                continue
            out.append((name, kwargs))
        return out

    def _validate_plan(self, plan: List[Tuple[str, dict]]) -> bool:
        """Check every primitive is in the whitelist + required args present."""
        if not plan:
            return False
        for name, kwargs in plan:
            spec = PRIMITIVE_REGISTRY.get(name)
            if spec is None:
                logger.warning("plan validate: unknown primitive %r", name)
                return False
            for req in spec["required_args"]:
                if req not in kwargs:
                    logger.warning("plan validate: %s missing required arg %s",
                                   name, req)
                    return False
        return True

    # ------------------------------------------------------------------
    # Execute primitives
    # ------------------------------------------------------------------

    def _exec_primitive(self, name: str, kwargs: dict) -> Tuple[bool, str]:
        """Look up primitive on the primitives layer, call it, normalize return."""
        if name not in PRIMITIVE_REGISTRY:
            return False, f"unknown primitive: {name}"
        method = getattr(self.p, name, None)
        if method is None or not callable(method):
            return False, f"primitive not implemented on layer 2: {name}"

        # Coerce kwargs (LLM sometimes sends list for tuple)
        try:
            kwargs = self._coerce_kwargs(name, kwargs)
        except ValueError as e:
            return False, f"bad kwargs for {name}: {e}"

        try:
            result = method(**kwargs)
        except TypeError as e:
            return False, f"call signature mismatch for {name}: {e}"
        except Exception as e:
            logger.exception("primitive %s raised", name)
            return False, f"exception: {e}"

        # Normalize return
        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), str(result[1])
        if isinstance(result, bool):
            return result, ""
        return True, ""    # void primitives counted as success

    @staticmethod
    def _coerce_kwargs(name: str, kwargs: dict) -> dict:
        """Mild type coercion for LLM-generated kwargs."""
        spec = PRIMITIVE_REGISTRY.get(name, {})
        types = spec.get("arg_types", {})
        out = {}
        for k, v in kwargs.items():
            t = types.get(k)
            if t is None:
                out[k] = v
                continue
            try:
                if t is int and isinstance(v, (str, float)):
                    out[k] = int(v)
                elif t is str and not isinstance(v, str):
                    out[k] = str(v)
                elif t == (list, tuple):
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        out[k] = (float(v[0]), float(v[1]))
                    else:
                        raise ValueError(f"{k} must be 2-element list/tuple")
                else:
                    out[k] = v
            except (ValueError, TypeError) as e:
                raise ValueError(f"{k}={v!r}: {e}")
        return out

    # ------------------------------------------------------------------
    # Reflect — decide a recovery primitive on failure
    # ------------------------------------------------------------------

    def _reflect(self, task: str, plan, step_i: int,
                 failed_prim: str, info: str) -> Optional[Tuple[str, dict]]:
        """
        Pick a recovery primitive based on failure.

        Cheap rules first; LLM only if rules don't match.
        """
        info_lower = info.lower()

        # Rule 1: approach failed because target lost → try once more
        if failed_prim == "approach":
            if "target_lost" in info_lower or "not_found" in info_lower:
                # Already retried internally, but external retry from observe pose
                # might help if the arm bumped something
                return ("go_to_observe", {})
            return ("approach", {"target_name": "carrot", "max_iters": 2})

        # Rule 2: grasp failed after all retries → can't easily recover, abort
        if failed_prim == "grasp_with_retry":
            # Internal retries exhausted. Best we can do: open + go home.
            return None

        # Rule 3: transport / release failures → release + go home for safety
        if failed_prim in ("transport_to", "release"):
            return ("go_home", {})

        # Rule 4: gripper / observe / home failures → just go home and abort
        if failed_prim in ("open_gripper", "close_gripper", "go_to_observe", "go_home"):
            return None

        # Fallback: no recovery
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kwargs_summary(kw: dict) -> str:
    """Compact one-line summary of kwargs for logs."""
    if not kw:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in kw.items())
