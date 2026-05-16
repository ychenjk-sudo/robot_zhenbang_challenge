"""
Main Entry Point + Gradio Web UI (RL Edition)
The robot learns online from real owner rewards.
"""

import os
import sys
import yaml
import time
import logging
import threading
import argparse
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Global shared state for Gradio UI
# ---------------------------------------------------------------------------
ui_state: Dict = {
    "status": "未启动",
    "round": 0,
    "accuracy": 0.0,
    "eps": 0.4,
    "baseline": 0.0,
    "last_choice": "-",
    "last_reward": "-",
    "policy_probs": [0.5, 0.5],
}

game_instance: Optional["ZhenbangGame"] = None
robot_instance: Optional["RobotActor"] = None
vision_instance: Optional["VisionSystem"] = None
voice_instance: Optional["VoiceSystem"] = None
trainer_instance: Optional["StatsTracker"] = None
voice_input_instance: Optional["VoiceInput"] = None
detector_instance: Optional["ObjectDetector"] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs():
    Path("./checkpoints").mkdir(exist_ok=True)


def init_hardware(cfg: dict):
    global robot_instance, vision_instance, voice_instance, trainer_instance

    rc = cfg["robot"]
    from robot_actor import RobotActor

    robot_instance = RobotActor(
        robot_type=rc["type"],
        port=rc["port"],
        robot_id=rc.get("id", "robot"),
        smooth_factor=rc.get("smooth_factor", 0.12),
        step_delay=rc.get("step_delay", 0.05),
        poses=cfg["game"]["poses"],
    )
    logger.info("Robot actor ready (%s)", rc["type"])

    cc = cfg["camera"]
    from vision import VisionSystem

    vision_instance = VisionSystem(
        camera_index=cc["index"],
        width=cc["width"],
        height=cc["height"],
        fps=cc["fps"],
    )
    vision_instance.start()
    logger.info("Vision ready (cam %d)", cc["index"])

    vc = cfg.get("voice", {})
    from voice import VoiceSystem, SoundConfig

    voice_instance = VoiceSystem(
        SoundConfig(
            sounds_dir=vc.get("sounds_dir", "./sounds"),
            volume=vc.get("volume", 0.8),
            fallback_tts=vc.get("fallback_tts", False),
        )
    )
    voice_instance.start()
    logger.info("Voice ready")

    global voice_input_instance
    from voice_input import VoiceInput

    voice_input_instance = VoiceInput()
    logger.info("Voice input registered (model loads on first use)")

    # Lightweight stats tracker — no CLIP, no torch model, just numbers.
    # (Was REINFORCETrainer; replaced now that SmolVLA owns action gen.)
    lc = cfg.get("learner", {})
    from stats_tracker import StatsTracker, TrackerConfig

    trainer_instance = StatsTracker(
        TrackerConfig(
            baseline_decay=lc.get("baseline_decay", 0.9),
        )
    )
    logger.info("Stats tracker ready (no CLIP, just accuracy/reward bookkeeping).")

    global detector_instance
    from detector import ObjectDetector

    # Use the same prompts and display names as the trainer for visual consistency
    detector_instance = ObjectDetector(
        prompts=list(trainer_instance.cfg.target_prompts),
        names=TARGET_DISPLAY_NAMES,
        backend="auto",  # tries YOLO first, falls back to color segmentation
    )
    logger.info("Detector registered (backend=auto, loads on first detection)")


def connect_robot(calibrate: bool = False):
    global robot_instance, ui_state
    if robot_instance is None:
        return False
    try:
        robot_instance.connect(calibrate=calibrate)
        ui_state["status"] = f"已连接 ({robot_instance.robot_type})"
        return True
    except Exception as e:
        ui_state["status"] = f"连接失败: {e}"
        logger.exception("Robot connect failed")
        return False


def disconnect_all():
    global game_instance
    if game_instance:
        game_instance.stop_game()
        game_instance = None
    if robot_instance:
        robot_instance.disconnect()
    if vision_instance:
        vision_instance.stop()
    if voice_instance:
        voice_instance.stop()


# ---------------------------------------------------------------------------
# Game control
# ---------------------------------------------------------------------------


def start_game():
    global game_instance, ui_state
    if game_instance and game_instance._running:
        return "游戏已在运行中"
    if robot_instance is None or not robot_instance.connected:
        return "机器人未连接，请先点击'连接机器人'"

    from game import ZhenbangGame, GameConfig

    def on_state(state, extra):
        ui_state["status"] = state.name
        ui_state["round"] = extra.get("round", 0)
        ui_state["accuracy"] = extra.get("acc", 0)
        ui_state["eps"] = extra.get("eps", 0)
        ui_state["baseline"] = extra.get("baseline", 0)
        if extra.get("reward") is not None:
            ui_state["last_reward"] = extra["reward"]
        if extra.get("choice") is not None:
            ui_state["last_choice"] = extra["choice"]

    game_cfg = GameConfig(
        correct_reward=1.0,
        wrong_reward=-0.1,
        auto_advance=False,  # manual rounds so owner can swap items
        enable_fakeout=True,
    )
    # Phase B: try to load calibrated workspace map. Falls back to Phase A
    # (3-zone discrete poses) if calibration.npz is missing.
    workspace = None
    try:
        from kinematics import WorkspaceMap

        workspace = WorkspaceMap("calibration.npz")
        logger.info("Phase B active: %s", workspace)
    except FileNotFoundError:
        logger.info(
            "Phase A active: no calibration.npz found (run `python calibrate.py` to enable continuous IK)."
        )
    except Exception as e:
        logger.warning("Failed to load workspace: %s — staying in Phase A.", e)

    # Phase H: SmolVLA — disabled (superseded by Phase D Diffusion Policy).
    vla_runner = None

    # Phase D: try to load Diffusion Policy runner
    dp_runner = None
    dp_checkpoint = "/Users/chenyuying/Downloads/lerobot_repo/checkpoints/diffusion_zhenbang_v2/checkpoints/002000/pretrained_model"
    dp_stats = "/Users/chenyuying/Downloads/lerobot_repo/datasets/local/zhenbang_pickplace_dp_v3/meta/stats.json"
    try:
        from dp_runner import DPRunner

        dp_runner = DPRunner()
        if dp_runner.load_checkpoint(dp_checkpoint, stats_path=dp_stats):
            logger.info("Phase D active: Diffusion Policy loaded for learned motor control.")
        else:
            dp_runner = None
    except Exception as e:
        logger.warning("Phase D not available: %s", e)

    # Phase I: try to build CaP agent (Tool Use mode). Needs detector + workspace.
    cap_agent = None
    cap_codegen = None
    primitives = None
    try:
        if detector_instance is not None and workspace is not None:
            from primitives import CaPPrimitives
            from cap_agent import PickAgent

            primitives = CaPPrimitives(
                robot=robot_instance,
                vision=vision_instance,
                detector=detector_instance,
                workspace=workspace,
                observe_pose_name="center_look",
                dp_runner=dp_runner,
            )
            cap_agent = PickAgent(
                primitives=primitives,
                detector=detector_instance,
                use_llm=True,
            )
            logger.info("Phase I (Tool Use) active: CaP agent ready (use_llm=%s).", cap_agent.use_llm)
        else:
            logger.info("Phase I not available: need both detector and workspace.")
    except Exception as e:
        logger.warning("CaP agent build failed: %s", e)

    # Phase I-A: REAL Code-as-Policies — LLM writes Python in a sandbox.
    # Reuses the same primitives instance as cap_agent (just a different agent
    # head on top). Needs Qwen API to be useful (no hardcoded fallback).
    try:
        if primitives is not None:
            from cap_codegen import CodeAgent

            llm_provider = None
            if cap_agent is not None and cap_agent._llm_provider:
                llm_provider = cap_agent._llm_provider
            cap_codegen = CodeAgent(
                primitives=primitives,
                llm_provider_info=llm_provider,
                max_iters=100,
                timeout_sec=30.0,
            )
            logger.info(
                "Phase I-A (real CaP codegen) active: LLM provider=%s.",
                llm_provider[0] if llm_provider else "NONE",
            )
    except Exception as e:
        logger.warning("CodeAgent build failed: %s", e)

    game_instance = ZhenbangGame(
        robot_actor=robot_instance,
        vision=vision_instance,
        voice=voice_instance,
        trainer=trainer_instance,
        detector=detector_instance,
        workspace=workspace,
        vla_runner=vla_runner,
        cap_agent=cap_agent,
        cap_codegen=cap_codegen,
        config=game_cfg,
        state_callback=on_state,
    )
    game_instance.dp_runner = dp_runner  # for UI status access
    game_instance.start_game()
    return "🎮 游戏已启动！机器人正在从奖励中学习。"


def stop_game():
    global game_instance, ui_state
    if game_instance:
        game_instance.stop_game()
        game_instance = None
    ui_state["status"] = "已停止"
    return "已停止"


TARGET_NAME_TO_IDX = {"萝卜": 0, "纸巾": 1, "可乐": 2}
TARGET_DISPLAY_NAMES = ["🥕 萝卜", "🧻 纸巾", "🥤 可乐"]


def _bar(p: float, width: int = 12) -> str:
    n = max(0, min(width, int(round(p * width))))
    return "█" * n + "░" * (width - n)


def _format_detection_row(d: dict, target_idx: Optional[int]) -> str:
    """One bullet for a detected object, with bbox center and score."""
    x1, y1, x2, y2 = d["bbox"]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    is_target = target_idx is not None and d["name_idx"] == target_idx
    marker = "🎯" if is_target else "•"
    name = d["name"]
    score = d["score"]
    return f"{marker} **{name}**  {score * 100:.0f}%  · bbox 中心 ({cx},{cy})"


def _format_thinking_vla() -> str:
    """VLA-specific thinking panel — show instruction, step progress, last action."""
    from game import GameState

    g = game_instance
    state = g.state

    lines = ["### 🧠 SmolVLA 推理"]

    if state in (GameState.IDLE, GameState.PAUSED):
        if g.last_decision_text:
            lines.append(f"*上一轮指令:* `{g.vla_instruction}`")
            lines.append(f"*执行步数:* {g.vla_step_count} / {g.vla_total_steps}")
            if g.vla_last_action:
                lines.append("*最终关节命令:*")
                for k, v in g.vla_last_action.items():
                    lines.append(f"  • {k}: `{v:+.1f}°`")
            lines.append("")
            lines.append("*等待出题（语音或点开始下一轮）...*")
        else:
            lines.append("*等待出题...*")
        return "\n".join(lines)

    if g.vla_status == "preparing":
        lines.append("⚙️ *正在归位 (home pose)…*")
        return "\n".join(lines)

    # Executing or done
    progress_pct = (g.vla_step_count / max(g.vla_total_steps, 1)) * 100
    bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
    lines.append(f"🎯 指令: `{g.vla_instruction}`")
    lines.append("")
    lines.append(f"📊 进度: `{bar}` {g.vla_step_count}/{g.vla_total_steps}  ({progress_pct:.0f}%)")

    if g.vla_episode_start_t:
        elapsed = time.time() - g.vla_episode_start_t
        lines.append(f"⏱️ 已耗: {elapsed:.1f}s / 上限 {g.vla_max_seconds:.0f}s")

    if g.vla_last_action:
        lines.append("")
        lines.append("**最新 action (关节角)**:")
        for k, v in g.vla_last_action.items():
            lines.append(f"  • `{k}: {v:+.2f}`")

    if g.vla_status == "done":
        lines.append("")
        lines.append("✅ *Episode 结束，等反馈 ✅/❌*")

    return "\n".join(lines)


def _format_thinking_cap() -> str:
    """CaP-specific thinking panel — show plan + current step + step log."""
    from game import GameState

    g = game_instance
    state = g.state

    lines = ["### 🧠 CaP Agent 推理"]

    if state in (GameState.IDLE, GameState.PAUSED):
        if g.cap_last_outcome:
            lines.append(f"*上一轮任务:* `{g.cap_task}`")
            lines.append(
                f"*结果:* {'✅ ' + g.cap_last_outcome if g.cap_last_outcome == 'success' else '❌ ' + g.cap_last_outcome}"
            )
            if g.cap_plan:
                lines.append(f"*Plan:* {' → '.join(g.cap_plan)}")
            lines.append("")
            lines.append("*等待出题...*")
        else:
            lines.append("*等待出题（说目标 或 点开始）...*")
        return "\n".join(lines)

    if g.cap_status == "preparing":
        lines.append("⚙️ *归位准备中…*")
        return "\n".join(lines)

    lines.append(f"🎯 任务: `{g.cap_task}`")
    lines.append("")

    if g.cap_plan:
        lines.append("**Plan:**")
        for i, step in enumerate(g.cap_plan):
            marker = "▶️" if i == g.cap_current_step_idx - 1 else "  "
            lines.append(f"  {marker} {i + 1}. `{step}`")
        lines.append("")

    if g.cap_current_step:
        lines.append(f"📍 当前: {g.cap_current_step}")

    # Recent step log
    if len(g.cap_step_log) > 0:
        lines.append("")
        lines.append("**最近事件:**")
        for ts, step, info in list(g.cap_step_log)[-6:]:
            t = time.strftime("%H:%M:%S", time.localtime(ts))
            short_info = (info[:60] + "…") if info and len(info) > 60 else (info or "")
            lines.append(f"  • `{t}` **{step}** — {short_info}")

    if g.cap_episode_start_t:
        elapsed = time.time() - g.cap_episode_start_t
        lines.append("")
        lines.append(f"⏱️ {elapsed:.1f}s")

    if g.cap_status == "done":
        lines.append("")
        if g.cap_last_outcome == "success":
            lines.append("✅ *Agent 自报成功，等主人最终确认 ✅/❌*")
        else:
            lines.append("⚠️ *Agent 自报失败，请反馈 ✅/❌*")

    return "\n".join(lines)


def format_thinking_from_state(target_idx: Optional[int]) -> str:
    """Render the reasoning panel, driven by game state."""
    from game import GameState

    if game_instance is None:
        return "### 🧠 思维过程\n*等待启动游戏...*"

    # Pick panel based on active backend (force_backend or auto-resolved)
    backend = getattr(game_instance, "force_backend", "auto")
    if backend == "auto":
        if game_instance.cap_agent is not None:
            backend = "cap"
        elif game_instance.vla_runner is not None:
            backend = "vla"
        else:
            backend = "perception"

    if backend == "cap" and game_instance.cap_agent is not None:
        return _format_thinking_cap()
    if backend == "vla" and game_instance.vla_runner is not None:
        return _format_thinking_vla()

    # If VLA is active (legacy path for backward compat), render VLA panel
    if game_instance.vla_runner is not None and backend != "perception":
        return _format_thinking_vla()

    state = game_instance.state
    if state in (GameState.IDLE, GameState.PAUSED):
        if game_instance.last_decision_text:
            last_t = game_instance.last_target_idx
            target_label = TARGET_DISPLAY_NAMES[last_t] if 0 <= last_t < len(TARGET_DISPLAY_NAMES) else "?"
            lines = [
                "### 🧠 思维过程 (上一轮)",
                f"🎯 目标: **{target_label}**",
                "",
                "**👀 Qwen2.5-VL 检测结果**",
            ]
            for d in game_instance.last_detections or []:
                lines.append(_format_detection_row(d, last_t))
            if not game_instance.last_detections:
                lines.append("*没有任何检测结果*")
            lines.append("")
            lines.append(f"🤔 决策: {game_instance.last_decision_text}")
            lines.append("")
            lines.append("*出新一轮的题继续...*")
            return "\n".join(lines)
        return "### 🧠 思维过程\n*等待出题（语音说目标 或 点'开始下一轮'）...*"

    target_label = (
        TARGET_DISPLAY_NAMES[target_idx]
        if target_idx is not None and 0 <= target_idx < len(TARGET_DISPLAY_NAMES)
        else "?"
    )

    lines = ["### 🧠 思维过程", f"🎯 目标: **{target_label}**", ""]

    if state == GameState.OBSERVING:
        lines.append("👀 🤖 *正在用 Qwen2.5-VL 全景检测...*")
    else:
        lines.append("**👀 Qwen2.5-VL 检测结果**")
        for d in game_instance.last_detections or []:
            lines.append(_format_detection_row(d, target_idx))
        if not game_instance.last_detections:
            lines.append("*暂无*")

    if game_instance.last_decision_text:
        lines.append("")
        lines.append(f"🤔 决策: {game_instance.last_decision_text}")
    elif state == GameState.DECIDING:
        lines.append("")
        lines.append("🤔 *正在挑选目标 detection...*")

    if state == GameState.HIT_LEFT:
        lines.append("👋 拍左！")
    elif state == GameState.HIT_CENTER:
        lines.append("👋 拍中！")
    elif state == GameState.HIT_RIGHT:
        lines.append("👋 拍右！")
    elif state == GameState.NOT_FOUND:
        lines.append("🤷 *没看到目标，跳过本轮*")
    elif state == GameState.WAITING_FEEDBACK:
        lines.append("")
        lines.append("⏸️ *等主人反馈：对了 / 错了*")
    elif state == GameState.CORRECT:
        lines.append("🎉 *主人说真棒！*")
    elif state == GameState.WRONG:
        lines.append("😿 *主人说错了...*")

    return "\n".join(lines)


def quick_round(zh_target: str):
    """One-click preset task. Auto-connect + auto-start game if needed."""
    global robot_instance, game_instance
    msgs = []
    # Step 1: ensure robot is connected
    if robot_instance is not None and not getattr(robot_instance, "connected", False):
        if connect_robot():
            msgs.append("🔌 已连接")
        else:
            return "❌ 机器人连接失败, 请检查串口"
    # Step 2: ensure game loop running
    if game_instance is None or not getattr(game_instance, "_running", False):
        msg = start_game()
        msgs.append(f"▶️ {msg}")
        time.sleep(0.5)  # let game thread spin up
    # Step 3: trigger the round
    result = next_round(zh_target, "")
    msgs.append(result)
    return "\n".join(msgs)


def quick_round_text(target_text: str):
    """Same auto-flow but with free-text target."""
    global robot_instance, game_instance
    msgs = []
    if robot_instance is not None and not getattr(robot_instance, "connected", False):
        if connect_robot():
            msgs.append("🔌 已连接")
    if game_instance is None or not getattr(game_instance, "_running", False):
        msg = start_game()
        msgs.append(f"▶️ {msg}")
        time.sleep(0.5)
    result = next_round("萝卜", target_text)
    msgs.append(result)
    return "\n".join(msgs)


def set_episode_duration(seconds: float) -> str:
    """Update VLA max episode time."""
    if game_instance is None:
        return f"⚙️ 待游戏启动后生效（暂存为 {seconds:.0f}s）"
    game_instance.vla_max_seconds = float(seconds)
    return f"✓ Episode 时长设为 {seconds:.0f}s"


def set_action_smoothing(alpha: float) -> str:
    """Adjust EMA smoothing on VLA action output."""
    if game_instance is None or game_instance.vla_runner is None:
        return f"⚙️ 待 VLA 启动后生效（暂存 alpha={alpha:.2f}）"
    game_instance.vla_runner.action_ema_alpha = float(alpha)
    return f"✓ Action 平滑 alpha={alpha:.2f}"


def set_backend(backend: str) -> str:
    """Switch execution backend: 'auto' / 'codegen' / 'cap' / 'vla' / 'perception'."""
    if game_instance is None:
        return f"⚙️ 待游戏启动后生效（暂存 backend={backend}）"
    backend = (backend or "auto").lower().strip()
    if backend not in ("auto", "codegen", "cap", "vla", "perception"):
        return f"❌ 未知 backend: {backend}"
    game_instance.force_backend = backend

    if backend == "auto":
        if game_instance.cap_codegen is not None:
            eff = "CodeGen (Phase I-A, 原版 CaP)"
        elif game_instance.cap_agent is not None:
            eff = "Tool Use (Phase I)"
        elif game_instance.vla_runner is not None:
            eff = "SmolVLA (Phase H)"
        else:
            eff = "Perception (Phase B)"
        dp = " + DP" if (getattr(game_instance, "dp_runner", None) is not None) else ""
        return f"✓ Backend = auto → {eff}{dp}"
    return f"✓ Backend = {backend}"


def set_grasp_x_offset(offset: float) -> str:
    """
    Camera-gripper X offset (world coords [0,1]²).
    Negative = grasp shifted LEFT, positive = RIGHT.
    For carrot grasp: try -0.03 ~ -0.08 to grip carrot's left edge.
    """
    if game_instance is None or game_instance.cap_agent is None:
        return f"⚙️ 待 Phase I 启动后生效（暂存 grasp_x_offset={offset:+.3f}）"
    primitives_obj = game_instance.cap_agent.p
    primitives_obj.grasp_x_offset = float(offset)
    return f"✓ Grasp X offset = {offset:+.3f} (- 左 / + 右)"


def set_grasp_y_offset(offset: float) -> str:
    """Camera-gripper Y offset. Negative = grasp shifted toward base, positive = forward."""
    if game_instance is None or game_instance.cap_agent is None:
        return f"⚙️ 待 Phase I 启动后生效（暂存 grasp_y_offset={offset:+.3f}）"
    primitives_obj = game_instance.cap_agent.p
    primitives_obj.grasp_y_offset = float(offset)
    return f"✓ Grasp Y offset = {offset:+.3f}"


def set_lift_offset(offset: float) -> str:
    """
    Manually offset shoulder_lift to compensate model's depth bias.
    Positive value → arm descends MORE (closer to table) before gripping.
    Useful when the carrot is gripped 1-2cm above the actual object.
    """
    if game_instance is None or game_instance.vla_runner is None:
        return f"⚙️ 待 VLA 启动后生效（暂存 lift_offset={offset:+.1f}°）"
    # Index 1 in JOINT_NAMES = shoulder_lift
    game_instance.vla_runner.action_offset[1] = float(offset)
    return f"✓ Lift offset = {offset:+.1f}° (正值=多下降)"


def reconnect_robot_ui() -> str:
    """
    UI handler for the prominent '🔌 连接机器人' button. Robust to:
      - never-connected state (first click)
      - already-connected state (no-op)
      - stale/bad connection (close + retry)
    Returns a string for the status_box.
    """
    global robot_instance
    if robot_instance is None:
        return "❌ Robot 模块未初始化, 请重启 main.py"
    # Already connected? Just report — user clicked button anyway as a probe.
    if getattr(robot_instance, "connected", False):
        return f"✅ 已连接 ({robot_instance.robot_type}) — 如果动作异常可再点一次重连"
    # Try connect (or reconnect)
    try:
        robot_instance.connect(calibrate=False)
        ui_state["status"] = f"已连接 ({robot_instance.robot_type})"
        return f"✅ 连接成功 ({robot_instance.robot_type}) — 现在可以点任务按钮了"
    except Exception as e:
        ui_state["status"] = f"连接失败: {e}"
        logger.exception("Robot reconnect failed")
        return (
            f"❌ 连接失败: {e}\n"
            "排查: 1) USB 串口插好了吗  2) 端口被其他程序占用?  "
            "3) 设备名是否对 (config.yaml)"
        )


def force_home() -> str:
    """Cancel any ongoing motion + force return to home pose."""
    global robot_instance, game_instance
    try:
        if game_instance is not None:
            game_instance._stop_event.set()
            time.sleep(0.2)
            game_instance._stop_event.clear()
        if robot_instance is None:
            return "❌ 机器人未连接"
        home = robot_instance.poses.get("home")
        if home is None:
            return "❌ 没有 home pose"
        robot_instance.go_to_pose(home, duration=1.5, steps=30)
        return "🏠 已回到 home"
    except Exception as e:
        logger.exception("force_home failed")
        return f"❌ {e}"


def render_action_stream():
    """Plot last N actions as a multi-line chart (one line per joint)."""
    if game_instance is None or not game_instance.vla_action_history:
        return None
    import io

    history = list(game_instance.vla_action_history)
    if len(history) < 2:
        return None
    steps = [s for s, _ in history]
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for j in joint_names:
        ys = [a[j] for _, a in history]
        ax.plot(steps, ys, "-", linewidth=1.4, label=j)
    ax.set_xlabel("step")
    ax.set_ylabel("joint angle")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title(f"Action stream (last {len(history)} steps, current {steps[-1]})")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return np.array(plt.imread(buf))


def get_backend_status() -> str:
    """Show which backend is active."""
    if game_instance is None:
        return "### 🔌 当前 Backend\n*等待启动游戏...*"

    forced = getattr(game_instance, "force_backend", "auto")

    # Resolve which backend would actually run
    def _effective(forced):
        if forced == "codegen" and game_instance.cap_codegen:
            return "codegen"
        if forced == "cap" and game_instance.cap_agent:
            return "cap"
        if forced == "vla" and game_instance.vla_runner:
            return "vla"
        if forced == "perception":
            return "perception"
        if forced == "auto":
            # codegen only if LLM available (matches game.py dispatch)
            if game_instance.cap_codegen is not None and getattr(game_instance.cap_codegen, "llm", None):
                return "codegen"
            if game_instance.cap_agent is not None:
                return "cap"
            if game_instance.vla_runner is not None:
                return "vla"
            return "perception"
        return forced

    eff = _effective(forced)
    badge = f" (force=`{forced}`)" if forced != "auto" else ""

    # Check Phase D (Diffusion Policy) status
    dp_active = (
        game_instance.cap_agent
        and hasattr(game_instance, "dp_runner")
        and game_instance.dp_runner is not None
    )

    if eff == "codegen":
        llm = "ON" if (game_instance.cap_codegen and game_instance.cap_codegen.llm) else "OFF"
        return (
            f"### 🔌 当前 Backend: 🔴 **Phase I-A (CodeGen, 原版 CaP)**{badge}\n"
            f"- LLM 现场写 Python: {llm}\n"
            f"- 沙盒: AST 白名单 + 100 iter 上限 + 30s 超时\n"
            f"- API 暴露: detect / hover_above / descend_and_grasp / ...\n"
            f"- 最接近 Liang et al. 2022 论文实现\n"
            + (f"- **🔵 Phase D: Diffusion Policy 已加载 (motor control)**" if dp_active else "")
        )
    if eff == "cap":
        llm = (
            "ON"
            if (
                game_instance.cap_agent
                and game_instance.cap_agent.use_llm
                and game_instance.cap_agent._llm_provider
            )
            else "OFF (hardcoded plans)"
        )
        return (
            f"### 🔌 当前 Backend: 🟣 **Phase I (Tool Use)**{badge}\n"
            f"- LLM 选 primitive: {llm}\n"
            f"- Layer 2: 视觉伺服 + grasp_with_retry + verify_grasp\n"
            f"- 比 CodeGen 安全, 但不能现场发明新动作\n"
            + (f"- **🔵 Phase D: Diffusion Policy 已加载 (motor control)**" if dp_active else "")
        )
    if eff == "vla":
        return (
            f"### 🔌 当前 Backend: 🟢 **Phase H (SmolVLA)**{badge}\n"
            f"- Checkpoint: `step_006000` (loss 0.176)\n"
            f"- 推理设备: MPS\n"
            f"- 训练任务: pick_carrot ×17, push_cola ×14, pull_tissue ×15\n"
            f"- Episode 时长: {game_instance.vla_max_seconds:.0f}s @ "
            f"{game_instance.vla_control_hz:.0f}Hz"
            + (f"\n- **🔵 Phase D: Diffusion Policy 已加载 (motor control)**" if dp_active else "")
        )
    if game_instance.workspace is not None:
        base = (
            f"### 🔌 当前 Backend: 🟡 **Phase B (检测+IK规则)**{badge}\n"
            "VLA 没加载, 退回到 Qwen2.5-VL bbox + RBF 插值"
        )
    else:
        base = f"### 🔌 当前 Backend: 🟠 **Phase A (3 段离散)**{badge}"
    dp_line = f"\n- **🔵 Phase D: Diffusion Policy 已加载 (motor control)**" if dp_active else ""
    return base + dp_line


def next_round(target_name: str = "萝卜", target_text: str = ""):
    """
    Start the next round. If target_text is non-empty, use free-text mode
    (detector.detect_target — works for ANY object Qwen knows). Otherwise
    fall back to the 3-class preset matched by target_name.
    """
    global game_instance
    if game_instance is None:
        return "游戏未运行"

    text = (target_text or "").strip()
    if text:
        game_instance.set_target_text(text)
        label = f"🎯 自由目标: {text}"
    else:
        target_idx = TARGET_NAME_TO_IDX.get(target_name, 0)
        game_instance.set_target(target_idx)
        label = f"🎯 预设目标: {target_name}"

    msg = game_instance.resume_next_round()
    return f"{label} | {msg}"


def user_feedback(correct: bool):
    """Owner gives reward signal."""
    global game_instance, ui_state
    if game_instance is None:
        return "游戏未运行"
    game_instance.give_feedback(correct)
    ui_state["last_reward"] = "✅ 对了" if correct else "❌ 错了"
    return ui_state["last_reward"]


IDX_TO_TARGET_NAME = {v: k for k, v in TARGET_NAME_TO_IDX.items()}


def handle_voice(audio_path):
    """Push-to-talk handler: transcribe → match keyword → fire the right action."""
    global game_instance, voice_input_instance
    if not audio_path:
        return "🎤 (未收到音频)"
    if voice_input_instance is None:
        return "🎤 语音模块未初始化"
    try:
        text = voice_input_instance.transcribe(audio_path)
    except Exception as e:
        logger.exception("transcribe failed")
        return f"🎤 识别失败: {e}"

    if not text:
        return "🎤 *没听清，再试一次*"

    from voice_input import route
    from game import GameState

    awaiting = game_instance is not None and game_instance.state == GameState.WAITING_FEEDBACK
    kind, payload = route(text, awaiting_feedback=awaiting)

    header = f"🎤 听到：**{text}**"
    if kind == "target":
        target_name = IDX_TO_TARGET_NAME.get(payload, "萝卜")
        result = next_round(target_name, "")
        return f"{header}\n→ 出题：**{target_name}** ({result})"
    if kind == "unknown":
        # Open-vocab: route the raw transcript to free-text target detection
        text_target = (payload if isinstance(payload, str) else text).strip()
        if text_target:
            result = next_round("萝卜", text_target)
            return f"{header}\n→ 自由目标：**{text_target}** ({result})"
    if kind == "feedback":
        verdict = "对了 ✅" if payload else "错了 ❌"
        result = user_feedback(payload)
        return f"{header}\n→ 反馈：**{verdict}**"
    return f"{header}\n→ ⚠️ 没匹配上关键词（试试：萝卜 / 纸巾 / 可乐 / 真棒 / 错了）"


def swap_items():
    """Toggle ground truth side so robot cannot cheat by position."""
    global game_instance
    if game_instance is None:
        return "游戏未运行"
    new_side = "right" if game_instance.current_ground_truth == "left" else "left"
    game_instance.set_ground_truth(new_side)
    return f"🔄 已交换！现在萝卜在 {new_side} 侧。继续下一轮来测试泛化能力。"


def reset_learning():
    """Clear accuracy/reward stats. Detection / action models are not affected."""
    global trainer_instance
    if trainer_instance is None:
        return "推理器未初始化"
    trainer_instance.reset_stats()
    return "🧹 准确率统计已清空。"


def emergency_stop():
    global robot_instance, game_instance, ui_state
    if game_instance:
        game_instance.stop_game()
        game_instance = None
    if robot_instance:
        robot_instance.stop()
        robot_instance.emergency_home()
    ui_state["status"] = "紧急停止"
    return "🛑 已执行紧急停止并归位"


def calibrate_pose(pose_name: str):
    global robot_instance
    if robot_instance is None or not robot_instance.connected:
        return "机器人未连接"
    pose = robot_instance.poses.get(pose_name)
    if not pose:
        return f"未知姿态: {pose_name}"
    robot_instance.go_to_pose(pose, duration=1.2, steps=24)
    return f"已运动到姿态: {pose_name}"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def plot_learning_curve():
    """Generate accuracy-tracking image for Gradio (inference mode)."""
    global trainer_instance
    if trainer_instance is None or len(trainer_instance.reward_history) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "还没有数据\n开始游戏后显示准确率曲线", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return img

    rh = list(trainer_instance.reward_history)
    accs = []
    window = 20
    for i in range(1, len(rh) + 1):
        window_vals = rh[max(0, i - window) : i]
        accs.append(np.mean([1 if r > 0.5 else 0 for r in window_vals]))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(
        range(len(rh)), rh, c=["green" if r > 0.5 else "red" for r in rh], s=15, alpha=0.6, label="每轮反馈"
    )
    ax.plot(range(len(accs)), accs, color="blue", linewidth=2, label=f"准确率 (window={window})")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Reward / Accuracy")
    ax.set_title(f"识别+执行 命中率 | Rounds: {len(rh)} | Current: {accs[-1] * 100:.0f}%")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.3, 1.2)

    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return img


def update_status_text():
    global ui_state, trainer_instance
    lines = [
        f"状态: {ui_state['status']}",
        f"回合: {ui_state['round']}",
        f"最近20轮准确率: {ui_state['accuracy'] * 100:.0f}%",
        f"上一轮选择: {ui_state['last_choice']}",
        f"上一轮反馈: {ui_state['last_reward']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui():
    import gradio as gr

    css = """
    .big-btn { font-size: 1.1rem !important; padding: 10px 20px !important; }
    .status-box { font-family: monospace; white-space: pre-wrap; }
    .highlight { color: #e65100; font-weight: bold; }
    .prob-left { color: #ff9800; font-weight: bold; }
    .prob-right { color: #9c27b0; font-weight: bold; }
    """

    with gr.Blocks(title="🤖 真棒挑战 · VLM + CaP + DP", css=css) as demo:
        gr.Markdown("# 🤖 萝卜·纸巾·可乐 · 真棒挑战")
        gr.Markdown("""
        | 模块 | 方案 |
        |------|------|
        | 👁️ 感知 | Qwen2.5-VL (API) |
        | 🧠 决策 | CaP Code as Policy (LLM 写代码编排 primitive) |
        | 🦾 执行 | Diffusion Policy (本地 50M 参数, learned motor control) |
        """)

        with gr.Row():
            # Left column: Camera + probability overlay
            with gr.Column(scale=2):
                camera_feed = gr.Image(
                    label="📷 摄像头画面（实时显示策略概率）",
                    streaming=True,
                    height=400,
                )
                thinking_md = gr.Markdown(
                    "### 🧠 思维过程\n*等待启动...*",
                    elem_id="thinking-panel",
                )

            # Right column: Linear-flow controls
            with gr.Column(scale=1):
                # ───────── STEP 0: 连接 ─────────
                gr.Markdown("## 🔌 机器人连接")
                connect_main_btn = gr.Button(
                    "🔌 连接 / 重连机器人",
                    variant="primary",
                    elem_classes=["big-btn"],
                )

                # ───────── 执行后端选择 ─────────
                gr.Markdown("## 🧠 执行后端")
                backend_selector = gr.Radio(
                    choices=[
                        ("自动 (优先 CodeGen + DP)", "auto"),
                        ("🔴 CodeGen (LLM 写代码 + DP 执行)", "codegen"),
                        ("🟣 Tool Use (LLM 选 primitive + DP 执行)", "cap"),
                    ],
                    value="auto",
                    label="点哪个就用哪个 backend",
                )

                # ───────── STEP 1: 选任务 ─────────
                gr.Markdown("## 1️⃣ 选任务 (一键开始)")
                with gr.Row():
                    quick_carrot_btn = gr.Button("🥕 拿萝卜", variant="primary", elem_classes=["big-btn"])
                    quick_tissue_btn = gr.Button("🧻 抽纸巾", elem_classes=["big-btn"])
                    quick_cola_btn = gr.Button("🥤 推可乐", elem_classes=["big-btn"])

                target_text_input = gr.Textbox(
                    label="🆎 或自由输入任意物体 (按回车开始)",
                    placeholder="例: 苹果 / banana / 蓝色杯子",
                    value="",
                )
                target_selector = gr.Radio(  # hidden, kept for voice compat
                    choices=["萝卜", "纸巾", "可乐"],
                    value="萝卜",
                    visible=False,
                )

                # ───────── STEP 2: 反馈 ─────────
                gr.Markdown("## 2️⃣ 反馈结果")
                with gr.Row():
                    correct_btn = gr.Button("✅ 任务完成", variant="primary", elem_classes=["big-btn"])
                    wrong_btn = gr.Button("❌ 未完成", variant="stop", elem_classes=["big-btn"])

                # ───────── 状态 ─────────
                status_box = gr.Textbox(
                    label="📊 运行状态",
                    value="未启动 · 点上面任意任务按钮自动连接 + 开始",
                    interactive=False,
                    lines=4,
                    elem_classes=["status-box"],
                )

                # ───────── 紧急控制 ─────────
                with gr.Row():
                    emergency_btn = gr.Button("🛑 立即停止", variant="stop", elem_classes=["big-btn"])
                    home_btn = gr.Button("🏠 回 home", elem_classes=["big-btn"])

                # ───────── 高级 / 调试 (折叠) ─────────
                with gr.Accordion("⚙️ 高级设置", open=False):
                    gr.Markdown("**DP 推理时长**")
                    episode_duration = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=5,
                        step=0.5,
                        label="DP 闭环执行 (秒)",
                    )
                    gr.Markdown("**Action 平滑度**")
                    action_smooth_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="EMA alpha (1.0=无平滑反应快, 0.3=丝滑慢)",
                    )
                    gr.Markdown(
                        "**🪛 Lift offset (深度修正)**  \n"
                        "若机械臂总停在萝卜上方 1-2cm 不下降, 调正值;  \n"
                        "若撞到桌面, 调负值。仅作用在 shoulder_lift 维度。"
                    )
                    lift_offset_slider = gr.Slider(
                        minimum=-15.0,
                        maximum=15.0,
                        value=0.0,
                        step=0.5,
                        label="Lift offset (°): 正值=多下降, 负值=少下降",
                    )
                    gr.Markdown(
                        "**🎯 抓取位置偏移 (Phase I 用)**  \n"
                        "夹爪夹在物体正中容易把细长物体撞飞 (萝卜常见症状).  \n"
                        "稍微往侧边挪一点, 让夹爪夹住物体边缘.  \n"
                        "X: **负=左, 正=右**.  萝卜建议先试 -0.04 ~ -0.08."
                    )
                    grasp_x_offset_slider = gr.Slider(
                        minimum=-0.20,
                        maximum=0.20,
                        value=0.0,
                        step=0.01,
                        label="Grasp X offset (world): 负=左 / 正=右",
                    )
                    grasp_y_offset_slider = gr.Slider(
                        minimum=-0.20,
                        maximum=0.20,
                        value=0.0,
                        step=0.01,
                        label="Grasp Y offset (world): 负=后 / 正=前",
                    )
                    gr.Markdown("**🎤 语音 (可选)**")
                    voice_mic = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="说 萝卜/纸巾/可乐 出题, 真棒/错了 反馈",
                        streaming=False,
                    )
                    voice_log = gr.Markdown("🎤 *待命*")

                with gr.Accordion("🔧 手动连接 (一般不需要)", open=False):
                    connect_btn = gr.Button("🔌 单独连接机器人")
                    connect_out = gr.Textbox(label="连接状态", interactive=False)
                    with gr.Row():
                        start_btn = gr.Button("▶️ 启动游戏循环")
                        stop_btn = gr.Button("⏹️ 停止")
                    next_btn = gr.Button("⏭️ 旧版: 单独触发一轮 (radio 选)")

                # Hidden: rarely used reset
                reset_btn = gr.Button("🧹 清空统计", visible=False)
                model_out = gr.Textbox(visible=False)

        # Live action stream chart (Priority 2.2)
        with gr.Accordion("🎛️ 实时 Action 流 (本轮关节命令时序)", open=True):
            action_stream_plot = gr.Image(
                label="最近 60 步关节命令",
                height=240,
            )

        # Backend status (Priority 3.1)
        backend_status = gr.Markdown("### 🔌 当前 Backend\n*启动游戏后自动检测...*")

        # Accuracy curve
        gr.Markdown("## 📈 VLA 命中率")
        gr.Markdown("SmolVLA 实机执行成功率（你按 ✅/❌ 反馈累积）。loss 0.176 的模型预期 70-85% 命中。")
        learning_plot = gr.Image(label="每轮反馈 + 滑动命中率")

        # Training curve link (Priority 3.4)
        with gr.Accordion("📉 训练 Loss 曲线 (本次 SFT 全过程)", open=False):
            gr.Markdown("训练 6000 步, 起点 loss 0.59 → 终点 0.176. 曲线 PNG 在 `/tmp/loss_curve.png`.")
            train_curve_img = gr.Image(value="/tmp/loss_curve.png", label="SFT loss curve", height=400)

        # Debug accordion — only useful poses kept (home for safety, others are
        # legacy from Phase A/B and not used by VLA execution).
        with gr.Accordion("🔧 手动姿态测试 (调试用)", open=False):
            pose_selector = gr.Dropdown(
                choices=["home", "center_look", "celebrate", "sad"],
                value="home",
                label="选择姿态",
            )
            pose_btn = gr.Button("🦾 运动到该姿态")
            pose_out = gr.Textbox(label="姿态结果", interactive=False)

        # Event handlers
        # Prominent main connect button (top of right column)
        connect_main_btn.click(fn=reconnect_robot_ui, outputs=status_box)
        # Legacy connect (still inside the 手动连接 accordion, returns bool)
        connect_btn.click(fn=connect_robot, outputs=connect_out)
        start_btn.click(fn=start_game, outputs=status_box)
        stop_btn.click(fn=stop_game, outputs=status_box)
        next_btn.click(fn=next_round, inputs=[target_selector, target_text_input], outputs=status_box)
        # Quick task buttons — now auto-connect + auto-start the game loop
        quick_carrot_btn.click(fn=lambda: quick_round("萝卜"), outputs=status_box)
        quick_tissue_btn.click(fn=lambda: quick_round("纸巾"), outputs=status_box)
        quick_cola_btn.click(fn=lambda: quick_round("可乐"), outputs=status_box)
        # Free-text target: pressing Enter in textbox triggers the round
        target_text_input.submit(fn=quick_round_text, inputs=target_text_input, outputs=status_box)
        # Episode duration (Priority 2.1)
        episode_duration.change(fn=set_episode_duration, inputs=episode_duration, outputs=status_box)
        # Action smoothing slider
        action_smooth_slider.change(fn=set_action_smoothing, inputs=action_smooth_slider, outputs=status_box)
        # Lift offset slider — compensate model depth bias (1-2cm above carrot)
        lift_offset_slider.change(fn=set_lift_offset, inputs=lift_offset_slider, outputs=status_box)
        # Grasp offset sliders — shift grasp position to grip object edge, not center
        grasp_x_offset_slider.change(fn=set_grasp_x_offset, inputs=grasp_x_offset_slider, outputs=status_box)
        grasp_y_offset_slider.change(fn=set_grasp_y_offset, inputs=grasp_y_offset_slider, outputs=status_box)
        # Backend selector — switch between CaP / VLA / perception at runtime
        backend_selector.change(fn=set_backend, inputs=backend_selector, outputs=status_box)
        voice_mic.stop_recording(fn=handle_voice, inputs=voice_mic, outputs=voice_log)
        correct_btn.click(fn=lambda: user_feedback(True), outputs=status_box)
        wrong_btn.click(fn=lambda: user_feedback(False), outputs=status_box)
        emergency_btn.click(fn=emergency_stop, outputs=status_box)
        # Force home (Priority 3.2)
        home_btn.click(fn=force_home, outputs=status_box)
        reset_btn.click(fn=reset_learning, outputs=model_out)
        pose_btn.click(fn=calibrate_pose, inputs=pose_selector, outputs=pose_out)

        # Camera shows raw frame with bbox overlay from the latest round's detections.
        # Thinking panel reads from game state machine (perception happens once per round).
        def update_feed():
            global vision_instance, game_instance, detector_instance
            if vision_instance is None:
                return None, "### 🧠 思维过程\n*摄像头未连接*"
            frame = vision_instance.read_frame()
            if frame is None:
                return None, "### 🧠 思维过程\n*无画面*"

            target_idx = game_instance.current_target_idx if game_instance else None
            # Persist last round's detections on the live camera so user sees the bboxes
            if game_instance and game_instance.last_detections and detector_instance:
                try:
                    frame = detector_instance.annotate(
                        frame, game_instance.last_detections, target_idx=target_idx
                    )
                except Exception:
                    pass

            think_md = format_thinking_from_state(target_idx)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), think_md

        camera_timer = gr.Timer(0.15)
        camera_timer.tick(fn=update_feed, outputs=[camera_feed, thinking_md])

        # Auto-refresh status, learning curve, action stream, backend status
        def refresh_dashboard():
            status = update_status_text()
            plot = plot_learning_curve()
            stream = render_action_stream()
            backend = get_backend_status()
            return status, plot, stream, backend

        dashboard_timer = gr.Timer(1.0)
        dashboard_timer.tick(
            fn=refresh_dashboard,
            outputs=[status_box, learning_plot, action_stream_plot, backend_status],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Robot 真棒 Challenge - RL Edition")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--no-ui", action="store_true", help="Headless mode")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate on connect")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("Config not found: %s", args.config)
        sys.exit(1)
    cfg = load_config(args.config)
    logger.info("Config loaded.")

    ensure_dirs()
    init_hardware(cfg)

    if args.no_ui:
        logger.info("Headless mode. Connecting...")
        if connect_robot(calibrate=args.calibrate):
            start_game()
            try:
                while game_instance and game_instance._running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        disconnect_all()
        return

    logger.info("Starting UI on port %d...", args.port)
    demo = build_ui()

    def delayed_connect():
        time.sleep(1.5)
        connect_robot(calibrate=args.calibrate)

    threading.Thread(target=delayed_connect, daemon=True).start()

    try:
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
    except KeyboardInterrupt:
        pass
    finally:
        disconnect_all()


if __name__ == "__main__":
    main()
