"""
Phase I-A — REAL Code-as-Policies (Liang et al. 2022 style).

跟 cap_agent.py (Tool Use 模式) 的区别:
  - cap_agent.py: LLM 输出 JSON, 我们解析 → 调白名单 primitive
  - cap_codegen.py (本文件): LLM 输出 **可执行 Python 源码**, 沙盒里 exec()

为什么这是"原版":
  Liang et al. 2022 "Code as Policies for Embodied Control" 的核心思想是
  LLM **生成 Python 代码**, 包含 while/if/for, 实时调感知 + 控制 API.
  这能让 LLM 现场组合出 demo 里没出现过的动作模式 (例如根据物体姿态
  动态选择抓取角度).

安全措施 (5 层):
  1. AST 白名单 — 只允许 if/for/while/赋值/函数调用, 禁止 import/exec/eval
  2. 名字黑名单 — 屏蔽 __dunder__, eval, exec, open, ...
  3. Restricted builtins — globals 只暴露 True/False/None/len/range/abs/...
  4. 循环计数器 — AST 注入, 超过 max_iters 抛异常
  5. 墙钟超时 — 每次 loop tick 检查 deadline

LLM 调用的 API 是 RobotAPI, 它是 cap_primitives 上面的安全 wrapper:
  - 所有运动函数都经过 workspace 边界检查
  - 没有底层电机访问
  - 失败 / 越界 → 返回 False, 不会抛异常
"""
from __future__ import annotations
import ast
import logging
import os
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ===========================================================================
# RobotAPI — LLM 生成的代码能用的所有函数
# ===========================================================================

class RobotAPI:
    """
    LLM 生成代码访问机器人的唯一接口. 任何危险操作都不可达.
    所有方法返回 bool 或安全数据, 不抛异常.
    """

    # World 坐标预设 (LLM 可以直接读)
    LEFT_DROP   = (0.15, 0.55)
    CENTER_DROP = (0.50, 0.55)
    RIGHT_DROP  = (0.85, 0.55)

    def __init__(self, primitives):
        """primitives: CaPPrimitives 实例 (复用 Tool Use 的 Layer 2)"""
        self.p = primitives
        self.log_buffer: list = []

    # ---------- 感知 ----------
    def detect(self, target_name: str) -> Optional[dict]:
        """
        定位目标, 返回 {'world_xy': (x, y), 'score': float} 或 None.
        会先回到 observe pose 再检测.
        """
        try:
            ok, _ = self.p.go_to_observe()
            if not ok:
                return None
            result = self.p._detect_target_pixel(target_name)
            if result is None:
                return None
            cx, cy, det = result
            world = self.p._pixel_to_world_safe(cx, cy)
            if world is None:
                return None
            return {"world_xy": world, "score": float(det.get("score", 0.9))}
        except Exception as e:
            self.log(f"detect error: {e}")
            return None

    def locate(self, target_name: str) -> Optional[Tuple[float, float]]:
        """detect 的简化版, 只返回 (wx, wy) 或 None."""
        d = self.detect(target_name)
        return d["world_xy"] if d else None

    def is_holding(self) -> bool:
        """夹爪角度是否在 holding 区间. Logs the actual angle for debugging."""
        try:
            from primitives import GRIP_HOLDING_MIN, GRIP_HOLDING_MAX
            self.p.robot._sync_joints_from_robot()
            g = float(self.p.robot.current_joints.get("gripper", 0.0))
            held = GRIP_HOLDING_MIN < g < GRIP_HOLDING_MAX
            self.log(f"is_holding: grip_angle={g:.1f} → "
                     f"{'YES' if held else 'NO'} "
                     f"(threshold {GRIP_HOLDING_MIN:.0f}..{GRIP_HOLDING_MAX:.0f})")
            return held
        except Exception as e:
            self.log(f"is_holding error: {e}")
            return False

    # ---------- 运动 ----------
    def hover_above(self, wx: float, wy: float) -> bool:
        """移动到 (wx, wy) 正上方 hover 高度, 夹爪打开. 越界返回 False."""
        try:
            wx, wy = float(wx), float(wy)
        except (TypeError, ValueError):
            return False
        if not self.p.workspace.in_bounds(wx, wy, margin=0.1):
            self.log(f"hover_above out of bounds: ({wx:.2f},{wy:.2f})")
            return False
        try:
            hover = self.p.workspace.joints_at(wx, wy, "hover")
            self.p._go_pose_with_gripper(hover, gripper_override=5.0,
                                          duration=0.8, steps=16)
            time.sleep(0.2)
            return True
        except Exception as e:
            self.log(f"hover_above error: {e}")
            return False

    def descend_and_grasp(self, wx: float, wy: float,
                           depth_offset: float = 0.0) -> bool:
        """
        下降到 grab 高度 + 闭夹爪. depth_offset 是 shoulder_lift 偏置(度,
        正值下降更多, 用于深度补偿). 返回是否真的抓住物体.

        会自动应用 camera-gripper 偏移 (primitives.grasp_x/y_offset).
        """
        try:
            wx, wy = float(wx), float(wy)
            depth_offset = float(depth_offset)
        except (TypeError, ValueError):
            return False
        # Apply camera-gripper offset (same as grasp_with_retry)
        wx_orig, wy_orig = wx, wy
        wx, wy = self.p.apply_grasp_offset(wx, wy)
        self.log(f"descend_and_grasp: target=({wx_orig:.3f},{wy_orig:.3f}) "
                 f"+offset=({self.p.grasp_x_offset:+.3f},{self.p.grasp_y_offset:+.3f}) "
                 f"→ ({wx:.3f},{wy:.3f})")
        if not self.p.workspace.in_bounds(wx, wy, margin=0.1):
            self.log(f"descend out of bounds: ({wx:.2f},{wy:.2f})")
            return False
        try:
            from primitives import GRIP_OPEN_VALUE, GRIP_CLOSE_VALUE
            hover = dict(self.p.workspace.joints_at(wx, wy, "hover"))
            hit = dict(self.p.workspace.joints_at(wx, wy, "hit"))
            if depth_offset != 0.0:
                offset = max(-10.0, min(10.0, depth_offset))
                hit["shoulder_lift"] = hit.get("shoulder_lift", 0.0) + offset

            # Phase 1: open gripper FULLY before any descent
            self.p._set_gripper(GRIP_OPEN_VALUE, duration=0.6)
            time.sleep(0.4)

            # Phase 2: hover above target (gripper still open)
            self.p._go_pose_with_gripper(hover, gripper_override=GRIP_OPEN_VALUE,
                                          duration=0.9, steps=18)
            time.sleep(0.3)

            # Phase 3: slow vertical descent
            self.p._go_pose_with_gripper(hit, gripper_override=GRIP_OPEN_VALUE,
                                          duration=1.4, steps=28)
            time.sleep(0.3)

            # Phase 4: close
            self.p._set_gripper(GRIP_CLOSE_VALUE, duration=0.6)
            time.sleep(0.6)
            return self.is_holding()
        except Exception as e:
            self.log(f"descend_and_grasp error: {e}")
            return False

    def grasp_with_retry(self, target_name: str = "carrot",
                          max_attempts: int = 3) -> bool:
        """
        高级抓取: 自动重新定位目标 + 微抖动 + 深度补偿 + 视觉验证, 最多重试
        max_attempts 次. 推荐优先用这个, 不要 descend_and_grasp 单次硬抓.
        返回是否真的抓住 (基于夹爪角度 + 视觉重检).
        """
        try:
            ok, info = self.p.grasp_with_retry(
                target_name=str(target_name),
                max_attempts=int(max_attempts),
            )
            self.log(f"grasp_with_retry → {ok} | {info}")
            return ok
        except Exception as e:
            self.log(f"grasp_with_retry error: {e}")
            return False

    def transport_to(self, wx: float, wy: float) -> bool:
        """携物移动到 (wx, wy). 保持当前夹爪状态."""
        try:
            ok, _ = self.p.transport_to((float(wx), float(wy)))
            return ok
        except Exception as e:
            self.log(f"transport_to error: {e}")
            return False

    def open_gripper(self) -> None:
        try:
            self.p.open_gripper()
        except Exception as e:
            self.log(f"open_gripper error: {e}")

    def close_gripper(self) -> None:
        try:
            self.p.close_gripper()
        except Exception as e:
            self.log(f"close_gripper error: {e}")

    def go_home(self) -> bool:
        try:
            ok, _ = self.p.go_home()
            return ok
        except Exception as e:
            self.log(f"go_home error: {e}")
            return False

    # ---------- 工具 ----------
    def workspace_in_bounds(self, wx: float, wy: float) -> bool:
        try:
            return self.p.workspace.in_bounds(float(wx), float(wy))
        except Exception:
            return False

    def sleep(self, seconds: float) -> None:
        """安全 sleep, 上限 2 秒 (防止 LLM 写 sleep(99999))."""
        try:
            s = float(seconds)
            time.sleep(min(max(s, 0.0), 2.0))
        except (TypeError, ValueError):
            pass

    def log(self, message: str) -> None:
        """LLM 代码里的 log/print 都走这里."""
        s = f"[code] {message}"
        logger.info(s)
        self.log_buffer.append((time.time(), str(message)))
        if len(self.log_buffer) > 200:
            self.log_buffer = self.log_buffer[-200:]


# ===========================================================================
# Sandbox — AST 白名单 + 受限执行
# ===========================================================================

class SandboxError(Exception):
    """LLM 代码违反安全策略, 或资源超限."""


# 允许的 AST 节点类型. 任何其他类型都被拒.
_ALLOWED_NODES = frozenset({
    ast.Module, ast.FunctionDef, ast.Return,
    ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Name, ast.Load, ast.Store, ast.Del,
    ast.Constant,
    ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Expr, ast.Call, ast.Attribute, ast.Subscript,
    ast.Index, ast.Slice,
    ast.If, ast.IfExp,
    ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.And, ast.Or, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.USub, ast.UAdd,
    ast.keyword, ast.arguments, ast.arg,
    ast.Try, ast.ExceptHandler,
    ast.JoinedStr, ast.FormattedValue,   # f-strings
    ast.Starred,                          # *args / *unpack in calls
})

_FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars",
    "__import__", "import",
    "getattr", "setattr", "delattr", "hasattr",
    "type", "object", "memoryview",
    "id", "hash", "breakpoint", "exit", "quit", "help",
    "__builtins__", "__name__", "__file__",
})


def validate_ast(code: str) -> None:
    """Walk the AST, reject forbidden constructs. Raises SandboxError."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}")

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_NODES:
            raise SandboxError(f"forbidden AST node: {node_type.__name__}")
        # Reject all import statements
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxError("imports not allowed")
        # Reject forbidden names
        if isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise SandboxError(f"forbidden name: {node.id}")
            if node.id.startswith("__"):
                raise SandboxError(f"forbidden dunder name: {node.id}")
        # Reject dunder attribute access (e.g. obj.__class__.__bases__)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _FORBIDDEN_NAMES:
                raise SandboxError(f"forbidden attribute: {node.attr}")


class _IterCounter:
    """Per-execution iteration counter. AST transform injects calls to tick()."""
    def __init__(self, limit: int, deadline: float):
        self.count = 0
        self.limit = limit
        self.deadline = deadline

    def tick(self):
        self.count += 1
        if self.count > self.limit:
            raise SandboxError(f"iteration limit exceeded ({self.limit})")
        if time.time() > self.deadline:
            raise SandboxError("timeout exceeded")


class _LoopGuardInjector(ast.NodeTransformer):
    """Inject _iter_counter.tick() at top of every for/while body."""
    def _make_tick(self):
        return ast.Expr(value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="_iter_counter", ctx=ast.Load()),
                attr="tick", ctx=ast.Load()),
            args=[], keywords=[]))

    def visit_For(self, node):
        self.generic_visit(node)
        node.body = [self._make_tick()] + node.body
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node.body = [self._make_tick()] + node.body
        return node


def execute_in_sandbox(code: str, api: RobotAPI,
                        max_iters: int = 100,
                        timeout_sec: float = 30.0) -> Tuple[bool, str]:
    """
    Validate + execute LLM code in sandbox. Returns (success, info).

    Pipeline:
      1. AST validation (reject forbidden constructs)
      2. AST transform: inject iter counter into all loops
      3. Compile + exec the def
      4. Call task() with the iter counter live
    """
    # 1. Validate
    try:
        validate_ast(code)
    except SandboxError as e:
        return False, f"validation failed: {e}"

    # 2. Transform AST (inject iteration guards)
    tree = ast.parse(code, mode="exec")
    tree = _LoopGuardInjector().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        compiled = compile(tree, "<llm-code>", "exec")
    except SyntaxError as e:
        return False, f"compile error: {e}"

    # 3. Restricted builtins
    safe_builtins = {
        "True": True, "False": False, "None": None,
        "len": len, "range": range,
        "abs": abs, "min": min, "max": max, "round": round,
        "int": int, "float": float, "str": str, "bool": bool,
        "tuple": tuple, "list": list, "dict": dict,
        "isinstance": isinstance, "sum": sum, "any": any, "all": all,
        "print": api.log,    # print() goes to api log
        "enumerate": enumerate, "zip": zip, "reversed": reversed,
    }

    deadline = time.time() + timeout_sec
    iter_counter = _IterCounter(limit=max_iters, deadline=deadline)
    sandbox_globals = {
        "__builtins__": safe_builtins,
        "api": api,
        "_iter_counter": iter_counter,
    }

    # 4. Define task()
    try:
        exec(compiled, sandbox_globals)
    except SandboxError as e:
        return False, f"sandbox: {e}"
    except Exception as e:
        return False, f"def error: {type(e).__name__}: {e}"

    task_fn = sandbox_globals.get("task")
    if task_fn is None or not callable(task_fn):
        return False, "no `task()` function defined"

    # 5. Run task() — deadline tick also covers non-loop code (kind of)
    try:
        result = task_fn()
    except SandboxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"runtime error: {type(e).__name__}: {e}"

    if not isinstance(result, bool):
        return False, f"task() must return bool, got {type(result).__name__}"
    return result, ("task succeeded" if result else "task returned False")


# ===========================================================================
# CodeAgent — Tool Use 的姐妹版, 但 LLM 输出 Python 而非 JSON
# ===========================================================================

CODE_PROMPT = """你是机器人编程助手. 写 Python 代码控制机械臂完成任务.

定义一个函数 `def task()`, 返回 True 表示成功, False 表示失败.

可用 API (只能通过 `api.` 调用, 不能 import 任何模块):

感知:
  api.detect(target_name) -> dict | None      # {'world_xy': (x, y), 'score': float}
  api.locate(target_name) -> tuple | None     # 简版, 返回 (wx, wy) 或 None
  api.is_holding() -> bool                    # 夹爪是否抓住物体

运动 (世界坐标 wx, wy ∈ [0, 1]):
  api.hover_above(wx, wy) -> bool             # 移到 (wx, wy) 上方 hover
  api.grasp_with_retry(target_name, max_attempts=3) -> bool  ⭐ 推荐!
      # 自动重新定位目标 + 微抖动 + 深度补偿 + 视觉验证.
      # 内部已经包含重试逻辑, 比 descend_and_grasp 更稳.
      # 调用前需要先 api.hover_above() 到目标附近.
  api.descend_and_grasp(wx, wy, depth_offset=0.0) -> bool
      # 单次下降+闭夹爪. 不重试. 一般不直接用, 用 grasp_with_retry.
      # depth_offset 是 shoulder_lift 偏置(度), 正值下降更多.
  api.transport_to(wx, wy) -> bool            # 携物移到 (wx, wy)
  api.open_gripper()
  api.close_gripper()
  api.go_home() -> bool
  api.sleep(seconds)                          # 上限 2 秒

工具:
  api.workspace_in_bounds(wx, wy) -> bool
  api.log(message)

预设落点 (按任务自动选择):
  api.LEFT_DROP   = (0.15, 0.55)   ← 萝卜 / carrot 默认放这里
  api.CENTER_DROP = (0.50, 0.55)
  api.RIGHT_DROP  = (0.85, 0.55)

任务到位置的默认映射 (重要!):
  - "萝卜" / "carrot" → LEFT_DROP   (拿起萝卜放到左边)
  - "纸巾" / "tissue" → CENTER_DROP
  - "可乐" / "cola"   → RIGHT_DROP  (推/拿可乐到右边)
  如果任务字符串没明确说位置, 按上面映射. 如果明确说了 (例如 "把萝卜放到中间"),
  按指令.

最佳实践:
  1. 先 api.locate() 找目标
  2. 再 api.hover_above() 飞到上方
  3. 再 api.grasp_with_retry() 抓 (它内部已经处理重试 + 深度补偿)
  4. 然后 api.transport_to() 搬到落点
  5. 最后 api.open_gripper() + api.go_home()
  不要直接调 api.descend_and_grasp(), 它没有重试.

硬性约束 (沙盒强制):
  - 不能 import
  - 循环最多 100 次迭代
  - 整个 task() 最多执行 30 秒
  - 没有 dunder (__xxx__) 访问

任务: {task}

输出仅 Python 代码, 无 markdown 围栏, 无解释. 必须包含 `def task():`."""


class CodeAgent:
    """
    原版 CaP — LLM 写 Python, 沙盒 exec.

    用法:
        agent = CodeAgent(primitives, llm_provider_info=qwen_provider)
        success, info = agent.run("拿萝卜放左边")
        # agent.last_code 里有这次生成的代码 (可用于 UI 显示)
    """

    def __init__(self, primitives, llm_provider_info=None,
                  max_iters: int = 100, timeout_sec: float = 30.0,
                  state_callback=None):
        self.api = RobotAPI(primitives)
        self.llm = llm_provider_info
        self.max_iters = max_iters
        self.timeout_sec = timeout_sec
        self.state_callback = state_callback

        self.last_task: str = ""
        self.last_code: str = ""
        self.last_outcome: str = ""
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def reset(self):
        self._cancel = False
        self.last_code = ""

    def _emit(self, step: str, info: str = ""):
        if self.state_callback:
            try:
                self.state_callback(step, info)
            except Exception as e:
                logger.warning("state_callback raised: %s", e)

    def run(self, task: str) -> Tuple[bool, str]:
        """Generate code → validate → sandbox-execute."""
        self.reset()
        self.last_task = task

        # Generate
        self._emit("codegen", "calling LLM...")
        code = self._generate_code(task)
        if not code:
            self.last_outcome = "failed"
            return False, "code generation returned empty"
        self.last_code = code
        self._emit("codegen", f"got {len(code)} chars of code")
        logger.info("[CodeAgent] generated code (%d chars):\n%s", len(code), code)

        # Execute
        self._emit("exec", "running in sandbox...")
        success, info = execute_in_sandbox(
            code, self.api,
            max_iters=self.max_iters, timeout_sec=self.timeout_sec,
        )
        self.last_outcome = "success" if success else "failed"
        self._emit("done", info)
        return success, info

    # --------------------------------------------------------------------
    def _generate_code(self, task: str) -> str:
        """Call LLM (Qwen text). Returns code string (cleaned of fences)."""
        if not self.llm:
            return ""
        try:
            from openai import OpenAI
        except ImportError:
            return ""
        provider, api_key, base_url, _vl_model = self.llm
        text_model = os.environ.get("ZHENBANG_CODE_MODEL", "qwen-plus")
        client = (OpenAI(api_key=api_key, base_url=base_url)
                  if base_url else OpenAI(api_key=api_key))
        # NOTE: CODE_PROMPT contains literal `{...}` (Python dict examples).
        # Using .format() would treat them as placeholders → KeyError.
        # Use .replace() to safely substitute only the {task} marker.
        prompt = CODE_PROMPT.replace("{task}", task)
        try:
            resp = client.chat.completions.create(
                model=text_model, max_tokens=1200, temperature=0.0,
                messages=[
                    {"role": "system",
                     "content": "你是机器人编程助手. 只输出 Python 代码, 不要解释."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return ""

        raw = resp.choices[0].message.content or ""
        return self._strip_code_fences(raw)

    @staticmethod
    def _strip_code_fences(raw: str) -> str:
        """Strip ```python ... ``` markdown fences."""
        raw = raw.strip()
        if raw.startswith("```"):
            # Cut first line (the ```python or ``` part)
            parts = raw.split("\n", 1)
            if len(parts) > 1:
                raw = parts[1]
            # Cut trailing ```
            raw = raw.rstrip()
            if raw.endswith("```"):
                raw = raw[:-3].rstrip()
        return raw.strip()
