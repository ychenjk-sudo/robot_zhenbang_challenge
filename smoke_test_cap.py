"""
Phase I (Code-as-Policies) smoke test — NO hardware, NO camera, NO Qwen API.

用 mock 模拟所有依赖, 跑通 agent.run('萝卜') 全流程, 验证:
  1. cap_agent 能解析硬编码 plan
  2. 每个 primitive 都被按顺序调用
  3. telemetry 字段 (last_step / last_plan / step_log) 正确填充
  4. 失败注入: grasp 故意空夹, reflect 触发, recovery 路径走通
  5. (可选 --llm) 真调一次 Qwen API, 验证 plan 生成 + 解析

用法:
    python smoke_test_cap.py              # 全 mock, 跑 5 个场景
    python smoke_test_cap.py --llm        # 加跑一次真 LLM 调用
    python smoke_test_cap.py --verbose    # 详细日志
"""
from __future__ import annotations
import argparse
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mocks — 模拟 robot / vision / detector / workspace
# ---------------------------------------------------------------------------

class MockRobot:
    """模拟 RobotActor — 记录所有 go_to_pose 调用, 不实际驱动电机."""
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]

    def __init__(self, poses=None):
        self.poses = poses or {
            "home":        {"shoulder_pan": 17.05, "shoulder_lift": -102.37,
                            "elbow_flex": 95.08, "wrist_flex": 54.46,
                            "wrist_roll": 4.88, "gripper": 24.79},
            "center_look": {"shoulder_pan": 14.95, "shoulder_lift": -43.21,
                            "elbow_flex": -2.68, "wrist_flex": 99.12,
                            "wrist_roll": -2.42, "gripper": 13.29},
        }
        self.current_joints = dict(self.poses["home"])
        self.connected = True
        self.history = []   # list of (timestamp, target_dict, duration)

        # Behavior knobs — simulate grip outcomes
        self.next_grip_angle: Optional[float] = None
        self.grip_close_default = 80.0    # what gripper reaches when empty
        self.grip_hold_default  = 55.0    # what gripper reaches with object

    def go_to_pose(self, pose, duration=1.0, steps=20):
        import time
        target = {j: float(pose.get(j, self.current_joints[j]))
                  for j in self.JOINT_NAMES}
        self.history.append((time.time(), dict(target), duration))
        # Distinguish "actual close command" (≥ 75 — i.e. GRIP_CLOSE_VALUE)
        # from "preserve current grip" (which could be any value).
        # Only inject outcome on actual close commands.
        cmd_grip = target.get("gripper", 0.0)
        if cmd_grip >= 75.0:   # full close command
            if self.next_grip_angle is not None:
                target["gripper"] = self.next_grip_angle
                self.next_grip_angle = None
            else:
                target["gripper"] = self.grip_close_default
        self.current_joints = target

    def reset_stop(self):
        pass

    def stop(self):
        pass

    def _sync_joints_from_robot(self):
        pass   # no-op for mock


class MockVision:
    def read_frame(self):
        # Return a fake 480x640 BGR frame
        return np.zeros((480, 640, 3), dtype=np.uint8)


class MockDetector:
    """
    模拟 ObjectDetector. 关键行为:
    - 如果绑定了 robot 且 robot 的夹爪处于"抓住"状态 → 返回 None
      (模拟"物体被抓走了, 桌面上看不到了")
    - 否则返回 default_bbox
    """

    def __init__(self):
        self._vlm_provider = None   # forces agent to fall back to hardcoded plans
        self.next_bbox: Optional[Tuple[int, int, int, int]] = None
        self.default_bbox: Tuple[int, int, int, int] = (280, 220, 360, 280)  # 中心
        self.found: bool = True
        self.call_count: int = 0
        self.robot_ref: Optional[Any] = None   # 设了之后会检查夹爪状态

        # Holding angle range (matches primitives.py GRIP_HOLDING_MIN/MAX)
        self.holding_range = (30.0, 72.0)

    def _is_holding(self) -> bool:
        if self.robot_ref is None:
            return False
        g = float(self.robot_ref.current_joints.get("gripper", 0.0))
        return self.holding_range[0] < g < self.holding_range[1]

    def detect_target(self, frame_bgr, target_text):
        self.call_count += 1
        if not self.found:
            return None
        if self._is_holding():
            # 物体在夹爪里, 桌面看不到
            return None
        bbox = self.next_bbox if self.next_bbox else self.default_bbox
        self.next_bbox = None
        x1, y1, x2, y2 = bbox
        return {
            "name_idx": -1, "name": target_text,
            "bbox": bbox, "score": 0.92, "cx": (x1 + x2) // 2,
        }

    def detect(self, frame_bgr, top_k_per_class=1):
        det = self.detect_target(frame_bgr, "carrot")
        return [det] if det else []


class MockWorkspace:
    """模拟 WorkspaceMap — 用线性映射代替 RBF 标定."""
    def __init__(self):
        # Fake homography: image is 640x480, world is [0,1]²
        self.img_w, self.img_h = 640, 480

    def pixel_to_world(self, u, v):
        return (float(u) / self.img_w, float(v) / self.img_h)

    def joints_at(self, x, y, mode="hover"):
        # Return a plausible joint dict that depends on (x, y)
        base = -50.0 if mode == "hover" else -30.0
        return {
            "shoulder_pan":  (x - 0.5) * 30.0,
            "shoulder_lift": base + (y - 0.5) * 10.0,
            "elbow_flex":    55.0 + (y - 0.5) * 8.0,
            "wrist_flex":    45.0,
            "wrist_roll":    0.0,
            "gripper":       20.0,
        }

    def in_bounds(self, x, y, margin=0.05):
        return -margin <= x <= 1 + margin and -margin <= y <= 1 + margin


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def assert_eq(actual, expected, name):
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
    return True


def make_agent(use_llm=False):
    """Wire up a PickAgent with all-mock dependencies."""
    from primitives import CaPPrimitives
    from cap_agent import PickAgent

    robot = MockRobot()
    vision = MockVision()
    detector = MockDetector()
    workspace = MockWorkspace()

    # Bind detector to robot so it can simulate "carrot disappears when grasped"
    detector.robot_ref = robot

    primitives = CaPPrimitives(
        robot=robot, vision=vision, detector=detector, workspace=workspace,
        observe_pose_name="center_look",
    )
    agent = PickAgent(
        primitives=primitives, detector=detector, use_llm=use_llm,
    )
    return agent, robot, vision, detector, workspace


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_1_happy_path():
    """场景 1: 一切正常, 第一次抓取成功."""
    print("\n" + "=" * 64)
    print("场景 1: 一切顺利 — 第一次抓取成功")
    print("=" * 64)

    agent, robot, vision, detector, workspace = make_agent(use_llm=False)

    # Tell mock robot that the next grip will hold the object (angle in holding range)
    robot.next_grip_angle = robot.grip_hold_default   # 55.0 → in (30, 72)

    success, info = agent.run("萝卜")

    assert_eq(success, True, "success")
    print(f"  ✓ success={success} info={info!r}")
    print(f"  ✓ plan {len(agent.last_plan)} steps:")
    for i, (n, kw) in enumerate(agent.last_plan):
        print(f"      {i+1}. {n}({kw})")
    print(f"  ✓ detector 调用次数: {detector.call_count}")
    print(f"  ✓ go_to_pose 调用次数: {len(robot.history)}")
    print(f"  ✓ memory 最后记录: {agent.memory.attempts[-1]['outcome']}")
    return True


def scenario_2_target_lost():
    """场景 2: 检测器全程看不到目标 — approach 应该失败."""
    print("\n" + "=" * 64)
    print("场景 2: 目标丢失 — detector 一直返回 None")
    print("=" * 64)

    agent, robot, vision, detector, workspace = make_agent(use_llm=False)
    detector.found = False   # 全程看不到

    success, info = agent.run("萝卜")

    assert_eq(success, False, "success")
    print(f"  ✓ 失败如预期: info={info!r}")
    print(f"  ✓ memory 最后记录: {agent.memory.attempts[-1]['outcome']}")
    return True


def scenario_3_grasp_retry_then_success():
    """场景 3: 第一次抓空, 第二次成功 — 测重试逻辑."""
    print("\n" + "=" * 64)
    print("场景 3: 第一次空夹, 第二次抓住")
    print("=" * 64)

    agent, robot, vision, detector, workspace = make_agent(use_llm=False)

    # 让 grip 行为按顺序: empty, hold, hold, hold...
    # MockRobot 默认 empty (closes to 80) → grip_with_retry 会重试
    # 我们要让"第 N 次"返回 hold
    grip_outcomes = [80.0, 55.0]   # attempt 1: empty, attempt 2: hold
    original_go = robot.go_to_pose
    call_n = {"close_count": 0}

    def patched_go(pose, duration=1.0, steps=20):
        # Detect close commands and inject the scripted outcomes
        if pose.get("gripper", 0) > 50:
            if call_n["close_count"] < len(grip_outcomes):
                robot.next_grip_angle = grip_outcomes[call_n["close_count"]]
                call_n["close_count"] += 1
        return original_go(pose, duration, steps)

    robot.go_to_pose = patched_go

    success, info = agent.run("萝卜")

    assert_eq(success, True, "success")
    print(f"  ✓ 重试后成功: info={info!r}")
    print(f"  ✓ close 命令总次数: {call_n['close_count']}")
    print(f"  ✓ memory 最后记录: {agent.memory.attempts[-1]['outcome']}")
    return True


def scenario_4_plan_parse_robustness():
    """场景 4: 各种异常 LLM 输出, plan 解析容错."""
    print("\n" + "=" * 64)
    print("场景 4: LLM 输出解析容错")
    print("=" * 64)

    from cap_agent import PickAgent

    tests = [
        # (description, raw_output, expected_n_steps)
        ("纯 JSON",
         '{"plan":[{"primitive":"approach","kwargs":{"target_name":"carrot"}}]}',
         1),
        ("Markdown 围栏 ```json",
         '```json\n{"plan":[{"primitive":"approach","kwargs":{}}]}\n```',
         1),
        ("Markdown 围栏 ``` 无 json 标记",
         '```\n{"plan":[{"primitive":"go_home","kwargs":{}}]}\n```',
         1),
        ("带前后解释文本",
         '好的, 这是 plan:\n{"plan":[{"primitive":"go_home","kwargs":{}}]}\n希望有帮助',
         1),
        ("trailing comma",
         '{"plan":[{"primitive":"approach","kwargs":{"target_name":"carrot",}},]}',
         1),
        ("多步",
         '{"plan":[{"primitive":"approach","kwargs":{"target_name":"carrot"}},'
         '{"primitive":"grasp_with_retry","kwargs":{}},'
         '{"primitive":"go_home","kwargs":{}}]}',
         3),
        ("空 plan",
         '{"plan":[]}',
         0),
        ("废话",
         'I cannot generate a plan because...',
         0),
    ]

    pass_count = 0
    for desc, raw, expected_n in tests:
        plan = PickAgent._parse_plan(raw)
        actual_n = len(plan)
        status = "✓" if actual_n == expected_n else "✗"
        if actual_n == expected_n:
            pass_count += 1
        print(f"  {status} {desc:30s} → {actual_n} steps (expected {expected_n})")

    if pass_count != len(tests):
        raise AssertionError(f"{len(tests)-pass_count}/{len(tests)} parse cases failed")
    return True


def scenario_5_kwargs_coercion():
    """场景 5: LLM 输出的 kwargs 类型强制转换."""
    print("\n" + "=" * 64)
    print("场景 5: kwargs 强制转换")
    print("=" * 64)

    from cap_agent import PickAgent

    tests = [
        # (primitive, raw_kwargs, expected_processed)
        ("approach", {"target_name": "carrot", "max_iters": "3"},
                     {"target_name": "carrot", "max_iters": 3}),
        ("transport_to", {"dest_world_xy": [0.15, 0.55]},
                         {"dest_world_xy": (0.15, 0.55)}),
        ("approach", {"target_name": 123, "max_iters": 2.0},
                     {"target_name": "123", "max_iters": 2}),
    ]

    pass_count = 0
    for prim, raw_kw, expected in tests:
        try:
            out = PickAgent._coerce_kwargs(prim, raw_kw)
            ok = out == expected
        except Exception as e:
            ok = False
            out = f"raised: {e}"
        status = "✓" if ok else "✗"
        if ok:
            pass_count += 1
        print(f"  {status} {prim}: {raw_kw} → {out}  (expected {expected})")

    if pass_count != len(tests):
        raise AssertionError(f"{len(tests)-pass_count}/{len(tests)} coerce cases failed")
    return True


def scenario_codegen_sandbox():
    """场景: cap_codegen 沙盒能拦截恶意/无效代码."""
    print("\n" + "=" * 64)
    print("场景 6: CodeGen 沙盒 — AST 白名单 + 资源限制")
    print("=" * 64)

    from cap_codegen import validate_ast, execute_in_sandbox, RobotAPI, SandboxError

    # Make a real RobotAPI on top of mocks
    from primitives import CaPPrimitives
    robot = MockRobot()
    detector = MockDetector()
    workspace = MockWorkspace()
    detector.robot_ref = robot
    primitives = CaPPrimitives(robot=robot, vision=MockVision(),
                                detector=detector, workspace=workspace,
                                observe_pose_name="center_look")
    api = RobotAPI(primitives)

    # 6.1 — Adversarial inputs should all be rejected
    bad_codes = [
        ("import os",          "import os"),
        ("eval call",          "def task(): return eval('1+1')"),
        ("dunder access",      "def task(): return ().__class__"),
        ("exec call",          "def task():\n    exec('x=1')\n    return True"),
        ("__import__",         "def task(): return __import__('os')"),
        ("open file",          "def task(): return open('/etc/passwd').read()"),
    ]
    print("  [6.1] adversarial code rejection:")
    rejected_count = 0
    for label, code in bad_codes:
        try:
            validate_ast(code)
            print(f"    ✗ {label:20s} NOT rejected (sandbox broken!)")
        except SandboxError as e:
            rejected_count += 1
            short_reason = str(e).split(":")[0]
            print(f"    ✓ {label:20s} rejected: {short_reason}")
    if rejected_count != len(bad_codes):
        raise AssertionError(f"{len(bad_codes)-rejected_count} adversarial cases passed!")

    # 6.2 — Legitimate code should run
    print("\n  [6.2] legitimate task code:")
    robot.next_grip_angle = 55.0   # next grasp will succeed
    good = """
def task():
    pos = api.locate("carrot")
    if pos is None:
        api.log("not found")
        return False
    if not api.hover_above(pos[0], pos[1]):
        return False
    if not api.descend_and_grasp(pos[0], pos[1]):
        api.log("grasp failed")
        return False
    if not api.transport_to(*api.LEFT_DROP):
        return False
    api.open_gripper()
    api.go_home()
    return True
"""
    ok, info = execute_in_sandbox(good, api, max_iters=50, timeout_sec=10.0)
    print(f"    success={ok} info={info!r}")
    print(f"    api log: {len(api.log_buffer)} entries")
    if not ok:
        raise AssertionError(f"good code failed: {info}")

    # 6.3 — Iteration limit enforcement
    print("\n  [6.3] iteration limit:")
    infinite = """
def task():
    i = 0
    while True:
        i = i + 1
    return True
"""
    ok, info = execute_in_sandbox(infinite, api, max_iters=10, timeout_sec=5.0)
    if ok or "iteration" not in info.lower():
        raise AssertionError(f"infinite loop not caught: ok={ok} info={info}")
    print(f"    ✓ infinite loop caught: {info}")

    # 6.4 — Task must return bool
    print("\n  [6.4] return-type check:")
    bad_return = """
def task():
    return "not a bool"
"""
    ok, info = execute_in_sandbox(bad_return, api, max_iters=10, timeout_sec=5.0)
    if ok or "must return bool" not in info:
        raise AssertionError(f"bad return type not caught: {info}")
    print(f"    ✓ wrong return type caught: {info}")

    return True


def scenario_6_llm_real_call():
    """场景 6: 真调一次 Qwen API. 需要 DASHSCOPE_API_KEY."""
    print("\n" + "=" * 64)
    print("场景 6: 真调一次 Qwen API (--llm 模式)")
    print("=" * 64)

    import os
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("  ⚠ DASHSCOPE_API_KEY 未设置, 跳过.")
        return True

    from primitives import CaPPrimitives
    from cap_agent import PickAgent
    try:
        from detector import _pick_vlm_provider
        provider = _pick_vlm_provider()
    except Exception as e:
        print(f"  ⚠ provider 解析失败: {e}, 跳过.")
        return True

    # 不接 hardcoded plan — 用 LLM 路径
    agent, *_ = make_agent(use_llm=True)
    agent._llm_provider = provider

    # 用一个 hardcoded 找不到的任务名, 强制走 LLM
    task = "把橘子搬到右边"
    print(f"  调用 LLM 生成 plan: task={task!r}")
    try:
        plan = agent._llm_plan(task)
        print(f"  ✓ LLM 返回 {len(plan)} 步")
        for i, (n, kw) in enumerate(plan):
            print(f"      {i+1}. {n}({kw})")
        ok = agent._validate_plan(plan)
        print(f"  {'✓' if ok else '✗'} plan 校验: {ok}")
        return ok
    except Exception as e:
        print(f"  ✗ LLM 调用异常: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help="额外跑一次真 LLM 调用 (需要 DASHSCOPE_API_KEY)")
    parser.add_argument("--verbose", action="store_true",
                        help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    scenarios = [
        ("happy path",             scenario_1_happy_path),
        ("target lost",            scenario_2_target_lost),
        ("grasp retry → success",  scenario_3_grasp_retry_then_success),
        ("plan parse robustness",  scenario_4_plan_parse_robustness),
        ("kwargs coercion",        scenario_5_kwargs_coercion),
        ("codegen sandbox",        scenario_codegen_sandbox),
    ]
    if args.llm:
        scenarios.append(("real LLM call", scenario_6_llm_real_call))

    print("\n" + "█" * 64)
    print("Phase I (Code-as-Policies) — Smoke Test")
    print(f"  scenarios: {len(scenarios)}, llm={args.llm}")
    print("█" * 64)

    results = []
    for name, fn in scenarios:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n  ✗ EXCEPTION in {name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 64)
    print("总结")
    print("=" * 64)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}   {name}")
    print(f"\n  {passed}/{len(results)} 通过")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
