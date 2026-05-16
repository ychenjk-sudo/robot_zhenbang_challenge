"""
Demo recording for SmolVLA pick-and-place training.

依赖 calibrate.py 一样的 leader+follower 配置 (config.yaml 的 robot.leader_port)。

用法:
    python record_demos.py
    python record_demos.py --fps 20 --max-duration 20 --output ./demos
    python record_demos.py --task pick_carrot   # 跳过菜单直接录某个任务

录制流程:
1. 连 leader+follower+摄像头，起 teleop loop
2. 菜单选任务 (拿萝卜/纸巾/可乐 等)
3. 把物体摆在桌上
4. 按 ENTER 开始录制 → 用 leader 臂演示完整 pick (或 pick+place)
5. 按 ENTER 停止 (或自动到 max_duration 停止)
6. 选保留/重录/删除
7. 回到菜单，下一条

输出目录结构:
    demos/
      episode_0000/
        frames/frame_0000.jpg ... frame_NNNN.jpg
        states.npy        (T, 6)  follower 关节位置
        actions.npy       (T, 6)  leader 命令位置
        meta.json         {task_name, instruction, fps, n_frames, ...}
      episode_0001/
        ...

后续用 convert_to_lerobot.py 转成 LeRobotDataset 喂 SmolVLA 训练。
"""
from __future__ import annotations
import argparse
import json
import logging
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("record")

# ---------------------------------------------------------------------------
# Task menu — extend here when you add new pick targets
# ---------------------------------------------------------------------------
TASKS = {
    # 实测 SO-101 夹爪能力调整：
    # 萝卜 — 能抓 ✓     纸巾 — 抽一张出来 ✓     可乐 — 夹得住举不起 → push 替代
    "1": ("pick_carrot",            "pick up the carrot",                                 "拿萝卜 (只 pick)"),
    "2": ("pick_place_carrot",      "pick up the carrot and place it on the left",       "拿萝卜从 右 → 左 (★)"),
    "3": ("pick_place_carrot_right","pick up the carrot and place it on the right",      "拿萝卜从 左 → 右 (★ 反向)"),
    "4": ("pull_tissue",            "pull a tissue from the roll",                        "抽纸巾"),
    "5": ("push_cola",              "push the cola can sideways",                         "推可乐"),
    "6": ("custom",                 "",                                                    "<自定义>"),
}


# ---------------------------------------------------------------------------
# Teleop thread + shared state (so recorder reads leader action without
# fighting the teleop loop for USB access)
# ---------------------------------------------------------------------------

class TeleopState:
    """Shared state between teleop thread and recorder."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_leader_action: Optional[dict] = None
        self.latest_follower_state: Optional[dict] = None


def _teleop_loop(leader, actor, state: TeleopState, stop: threading.Event,
                 hz: float = 50.0):
    """
    Continuously copy leader → follower, publish:
      - latest_leader_action  : what we COMMAND the follower to be
      - latest_follower_state : where the follower ACTUALLY is right now
    These differ by motor latency — that delta is the signal SmolVLA learns.
    """
    dt = 1.0 / hz
    while not stop.is_set():
        t0 = time.time()
        try:
            # 1) Get leader's commanded position
            action = leader.get_action()
            if action is not None:
                # 2) Forward to follower (this is the teleop)
                actor.robot.send_action(action)
                with state.lock:
                    state.latest_leader_action = dict(action)

            # 3) Independently read follower's TRUE current position
            #    (not just what we just commanded — actual motor encoder reading)
            try:
                obs = actor.robot.get_observation()
                if obs is not None:
                    for j in actor.JOINT_NAMES:
                        for k in (f"{j}.pos", j, f"observation.{j}.pos"):
                            if k in obs:
                                actor.current_joints[j] = float(obs[k])
                                break
                    with state.lock:
                        state.latest_follower_state = dict(actor.current_joints)
            except Exception:
                pass
        except Exception:
            pass
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ---------------------------------------------------------------------------
# Episode recorder
# ---------------------------------------------------------------------------

class EpisodeRecorder:
    def __init__(self, output_dir: str, fps: float = 20.0,
                 max_duration: float = 15.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_duration = max_duration
        self.frame_dt = 1.0 / fps

    def _next_episode_id(self) -> int:
        existing = sorted(self.output_dir.glob("episode_*"))
        if not existing:
            return 0
        try:
            return int(existing[-1].name.split("_")[1]) + 1
        except ValueError:
            return len(existing)

    def record(self, vision, actor, state: TeleopState,
               task_name: str, instruction: str) -> tuple[bool, int]:
        """
        Record one episode. Returns (saved, n_frames).
        Saved=False means the user discarded it.
        """
        ep_id = self._next_episode_id()
        ep_dir = self.output_dir / f"episode_{ep_id:04d}"
        frames_dir = ep_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        joint_names = list(actor.JOINT_NAMES)

        print(f"\n━━━━━━━━━ Episode {ep_id} | {task_name} ━━━━━━━━━")
        print(f"  指令: \"{instruction}\"")
        print(f"  最大时长: {self.max_duration:.0f}s   FPS: {self.fps:.0f}")
        print(f"  摄像头分辨率自动取 vision.read_frame() 输出")
        try:
            input("  ▶ 把物体摆好，握住 leader 臂，按 ENTER 开始录制… ")
        except (EOFError, KeyboardInterrupt):
            shutil.rmtree(ep_dir, ignore_errors=True)
            return False, 0

        # Countdown
        for c in (3, 2, 1):
            print(f"  {c}…", flush=True)
            time.sleep(0.5)
        print("  ● 录制中 — 按 ENTER 结束 (或到达最大时长自动停止)\n")

        # Background stdin watcher
        stop_flag = [False]

        def watch_enter():
            try:
                sys.stdin.readline()
            except Exception:
                pass
            stop_flag[0] = True

        watcher = threading.Thread(target=watch_enter, daemon=True)
        watcher.start()

        states, actions, frame_paths = [], [], []
        t_start = time.time()
        next_t = t_start

        last_log_t = t_start
        try:
            while not stop_flag[0]:
                now = time.time()
                if now - t_start >= self.max_duration:
                    print(f"  ⏱  达到最大时长 {self.max_duration:.0f}s，自动停止")
                    break
                if now < next_t:
                    time.sleep(min(0.005, next_t - now))
                    continue
                next_t += self.frame_dt

                frame = vision.read_frame() if vision else None
                if frame is None:
                    continue

                with state.lock:
                    follower_dict = state.latest_follower_state
                    leader_dict = state.latest_leader_action

                if follower_dict is None:
                    continue

                state_vec = np.array(
                    [float(follower_dict.get(j, 0.0)) for j in joint_names],
                    dtype=np.float32,
                )
                if leader_dict is not None:
                    action_vec = np.array(
                        [float(leader_dict.get(f"{j}.pos",
                                               leader_dict.get(j, state_vec[i])))
                         for i, j in enumerate(joint_names)],
                        dtype=np.float32,
                    )
                else:
                    action_vec = state_vec.copy()

                idx = len(states)
                path = frames_dir / f"frame_{idx:05d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_paths.append(path.name)
                states.append(state_vec)
                actions.append(action_vec)

                if now - last_log_t > 1.0:
                    print(f"    {len(states)} frames  ({(now-t_start):.1f}s)")
                    last_log_t = now
        except KeyboardInterrupt:
            print("\n  ⚠ Ctrl+C 中止")
            stop_flag[0] = True

        n = len(states)
        elapsed = time.time() - t_start
        print(f"\n  采集完成: {n} 帧 / {elapsed:.1f}s "
              f"(实际 {n / max(elapsed, 1e-6):.1f} FPS)")

        if n < 5:
            shutil.rmtree(ep_dir, ignore_errors=True)
            print("  ✗ 太短了，删除。")
            return False, n

        # Persist arrays + metadata
        np.save(ep_dir / "states.npy", np.stack(states))
        np.save(ep_dir / "actions.npy", np.stack(actions))
        meta = {
            "episode_id": ep_id,
            "task_name": task_name,
            "instruction": instruction,
            "fps": self.fps,
            "n_frames": n,
            "joint_names": joint_names,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        (ep_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # Keep / discard / replay
        try:
            choice = input(
                "  这条 episode 保留吗? (Y=保留 / n=删除 / r=删除并重录) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "y"

        if choice == "n":
            shutil.rmtree(ep_dir)
            print("  ✗ 已删除")
            return False, n
        if choice == "r":
            shutil.rmtree(ep_dir)
            print("  ↻ 重录")
            return self.record(vision, actor, state, task_name, instruction)

        print(f"  ✓ 保存到 {ep_dir}")
        return True, n


# ---------------------------------------------------------------------------
# Helpers (re-using bits from calibrate.py via local imports)
# ---------------------------------------------------------------------------

def _make_actor(cfg: dict):
    rc = cfg["robot"]
    from robot_actor import RobotActor
    return RobotActor(
        robot_type=rc["type"], port=rc["port"], robot_id=rc.get("id", "robot"),
        smooth_factor=rc.get("smooth_factor", 0.12),
        step_delay=rc.get("step_delay", 0.05),
        poses=cfg["game"]["poses"],
    )


def _connect_leader(leader_port: str, robot_id: str, robot_type: str = "so101"):
    from lerobot.teleoperators.so_leader import (
        SO101LeaderConfig, SO101Leader, SO100LeaderConfig, SO100Leader,
    )
    if robot_type == "so101":
        cfg = SO101LeaderConfig(port=leader_port, id=robot_id)
        leader = SO101Leader(cfg)
    else:
        cfg = SO100LeaderConfig(port=leader_port, id=robot_id)
        leader = SO100Leader(cfg)
    leader.connect(calibrate=False)
    return leader


def _menu_select_task() -> Optional[tuple]:
    print("\n──────── 选择任务 ────────")
    for k, (name, instr, zh) in TASKS.items():
        if name == "custom":
            print(f"  {k}. {zh}  (自己输入指令)")
        else:
            print(f"  {k}. {zh}  →  \"{instr}\"")
    print("  q. 退出录制")
    print("──────────────────────────")
    choice = input("> ").strip().lower()
    if choice == "q":
        return None
    if choice not in TASKS:
        print(f"  无效选择: {choice}")
        return _menu_select_task()
    name, instr, zh = TASKS[choice]
    if name == "custom":
        zh = input("  自定义任务名 (英文/拼音, 比如 stack_can): ").strip() or "custom"
        instr = input("  自定义英文指令 (会作为 SmolVLA 的 prompt): ").strip()
        if not instr:
            instr = f"perform {zh}"
        name = zh
    return name, instr


def _print_summary(out_dir: Path):
    eps = sorted(out_dir.glob("episode_*"))
    print(f"\n=== 数据集汇总 ({out_dir}) ===")
    by_task: dict[str, int] = {}
    total_frames = 0
    for ep in eps:
        try:
            meta = json.loads((ep / "meta.json").read_text())
            t = meta.get("task_name", "?")
            by_task[t] = by_task.get(t, 0) + 1
            total_frames += int(meta.get("n_frames", 0))
        except Exception:
            continue
    print(f"  总 episodes: {len(eps)}    总帧数: {total_frames}")
    for t, n in sorted(by_task.items()):
        print(f"    • {t}: {n} 条")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output", default="./demos")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--max-duration", type=float, default=15.0)
    ap.add_argument("--task", default=None,
                    help="跳过菜单直接录某个任务 (pick_carrot 等)")
    ap.add_argument("--instruction", default=None,
                    help="自定义指令文本")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rc = cfg["robot"]
    leader_port = rc.get("leader_port", "")
    if not leader_port:
        print("✗ config.yaml 里 robot.leader_port 没填。先填好再跑。")
        sys.exit(1)

    # Connect everything
    print("→ 连接 follower …")
    actor = _make_actor(cfg)
    actor.connect(calibrate=False)

    print(f"→ 连接 leader ({leader_port}) …")
    leader = _connect_leader(leader_port, rc.get("leader_id", "zhenbang_leader"),
                             rc.get("type", "so101"))

    print("→ 启动摄像头 …")
    from vision import VisionSystem
    cc = cfg["camera"]
    vision = VisionSystem(camera_index=cc["index"], width=cc["width"],
                          height=cc["height"], fps=cc["fps"])
    vision.start()

    # Move follower to a sensible start pose
    home = actor.poses.get("home")
    if home:
        print("→ follower 移到 home …")
        actor.go_to_pose(home, duration=1.5, steps=30)

    # Start teleop
    state = TeleopState()
    stop_event = threading.Event()
    teleop_thr = threading.Thread(
        target=_teleop_loop, args=(leader, actor, state, stop_event), daemon=True)
    teleop_thr.start()
    time.sleep(0.5)
    print("✓ Teleop 已启动 (leader → follower)\n")

    recorder = EpisodeRecorder(args.output, fps=args.fps,
                               max_duration=args.max_duration)
    saved_count = 0

    try:
        while True:
            if args.task:
                task_name, instruction = args.task, (
                    args.instruction
                    or next((i for n, i, _ in TASKS.values() if n == args.task),
                            f"do {args.task}"))
            else:
                pick = _menu_select_task()
                if pick is None:
                    break
                task_name, instruction = pick

            ok, n = recorder.record(vision, actor, state, task_name, instruction)
            if ok:
                saved_count += 1

            if args.task:
                # When invoked with --task, ask whether to record another of the
                # same kind or quit
                again = input("\n再录一条同样任务? (Y/n) > ").strip().lower()
                if again == "n":
                    break
    except KeyboardInterrupt:
        print("\n用户中止")
    finally:
        stop_event.set()
        teleop_thr.join(timeout=1.0)
        try: leader.disconnect()
        except Exception: pass
        try:
            if home:
                actor.go_to_pose(home, duration=1.5, steps=30)
            actor.disconnect()
        except Exception:
            pass
        try: vision.stop()
        except Exception: pass

    _print_summary(Path(args.output))
    print(f"\n本次录制保留 {saved_count} 条 episode。")


if __name__ == "__main__":
    main()
