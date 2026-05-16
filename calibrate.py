"""
Phase B 标定脚本：摄像头 homography + 机械臂工作空间采样。

用法:
    python calibrate.py                          # 完整流程，自动检测 leader_port
    python calibrate.py --leader-port /dev/...   # 显式指定 leader 端口
    python calibrate.py --keyboard               # 强制软件键盘模式（无 leader 时）
    python calibrate.py --camera-only / --robot-only
    python calibrate.py --grid 3                 # 3x3=9 点 (默认)，也可 --grid 2

输出:
    calibration.npz  ← 后续 detector / robot 都用这个文件

⚠️ 关键约束（摄像头装机械臂上必读）:
    标定摄像头时，机械臂必须在 `center_look` 姿势 (与游戏运行时观察的姿势一致)。
    脚本会自动把机械臂送到 center_look 再让你点击 4 个角。
    游戏运行时 game.py 也会先到 center_look 再拍照，所以 homography 永远有效。

两种采样模式：
  • TELEOP 模式 (推荐): 用 leader 臂控制 follower，自然顺畅
    需要在 config.yaml 填 robot.leader_port 或 --leader-port 指定
  • KEYBOARD 模式: 没 leader 臂时备选，用键盘逐关节微调
    --keyboard 强制启用
"""
from __future__ import annotations
import sys
import argparse
import logging
import time
from pathlib import Path
import numpy as np
import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("calibrate")


# ---------------------------------------------------------------------------
# Camera homography  (4-click)
# ---------------------------------------------------------------------------

def calibrate_camera(camera_index: int, width: int, height: int) -> np.ndarray:
    """
    User clicks 4 corners (TL, TR, BR, BL) of the calibration rectangle in
    the camera view. Returns 3x3 homography H mapping pixel (u,v,1) → world (x,y,1)
    where world coords are the unit square [0,1]^2.
    """
    print("\n=== 摄像头标定 ===")
    print("画面里依次点击标定矩形的 4 个角：")
    print("  1) 左上  2) 右上  3) 右下  4) 左下")
    print("  按 'r' 重置  按 'q' 取消  按 ENTER/SPACE 确认")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    points: list = []
    labels = ["TL (左上)", "TR (右上)", "BR (右下)", "BL (左下)"]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  点 {len(points)}/4: {labels[len(points)-1]} = ({x}, {y})")

    win = "Camera Calibration"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        H, W = frame.shape[:2]
        disp = frame.copy()

        for i, p in enumerate(points):
            cv2.circle(disp, p, 8, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(disp, points[i-1], points[i], (0, 255, 255), 2)
            if len(points) == 4:
                cv2.line(disp, points[3], points[0], (0, 255, 255), 2)

        instr = (f"Click corner {len(points)+1}: {labels[len(points)]}"
                 if len(points) < 4 else "All 4 corners set. Press ENTER to confirm, 'r' reset, 'q' cancel.")
        cv2.rectangle(disp, (0, 0), (W, 35), (30, 30, 30), -1)
        cv2.putText(disp, instr, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("用户取消摄像头标定")
        elif key == ord('r'):
            points.clear()
            print("  重置，请重新点击 4 个角")
        elif key in (13, 32) and len(points) == 4:  # ENTER or SPACE
            break

    cap.release()
    cv2.destroyAllWindows()

    src = np.array(points, dtype=np.float32)
    # Unit square in world coords: TL=(0,0), TR=(1,0), BR=(1,1), BL=(0,1)
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H_mat, _ = cv2.findHomography(src, dst)
    print(f"\n摄像头 homography 计算完成。")
    return H_mat


# ---------------------------------------------------------------------------
# Robot workspace sampling
# ---------------------------------------------------------------------------

def grid_points(n: int) -> np.ndarray:
    """Return (n*n, 2) array of normalized grid points in [0, 1]²."""
    if n < 2:
        raise ValueError("grid size must be >= 2")
    coords = np.linspace(0.0, 1.0, n)
    pts = np.array([(x, y) for y in coords for x in coords])
    return pts


# ---------------------------------------------------------------------------
# Leader arm + teleop background thread
# ---------------------------------------------------------------------------

class LeaderArm:
    """Wraps lerobot SO-leader so we can run teleop in a background thread."""

    def __init__(self, port: str, robot_id: str = "zhenbang_leader",
                 robot_type: str = "so101"):
        from lerobot.teleoperators.so_leader import (
            SO101LeaderConfig, SO101Leader, SO100LeaderConfig, SO100Leader)
        if robot_type == "so101":
            cfg = SO101LeaderConfig(port=port, id=robot_id)
            self.leader = SO101Leader(cfg)
        else:
            cfg = SO100LeaderConfig(port=port, id=robot_id)
            self.leader = SO100Leader(cfg)
        self.connected = False

    def connect(self):
        self.leader.connect(calibrate=False)
        self.connected = True
        logger.info("Leader arm connected.")

    def disconnect(self):
        if self.connected:
            try: self.leader.disconnect()
            except Exception: pass
            self.connected = False

    def get_action(self):
        return self.leader.get_action()


def _start_teleop_thread(leader: "LeaderArm", actor) -> tuple:
    """Start background thread: leader.action → follower.send_action.
    Returns (thread, stop_event)."""
    import threading
    stop = threading.Event()

    def loop():
        # ~50 Hz
        while not stop.is_set():
            try:
                action = leader.get_action()
                if action is not None:
                    actor.robot.send_action(action)
                    # Mirror into actor.current_joints so snapshots see the right values
                    for j in actor.JOINT_NAMES:
                        for k in (f"{j}.pos", j):
                            if k in action:
                                actor.current_joints[j] = float(action[k])
                                break
            except Exception as e:
                # Don't spam logs every tick
                pass
            time.sleep(0.02)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    logger.info("Teleop loop running (leader → follower).")
    return t, stop


def _teleop_record(actor, prompt: str) -> dict:
    """In teleop mode: user moves leader, we just wait for ENTER and snapshot."""
    print("\n" + prompt)
    input("  按 ENTER 记录此姿势  >  ")
    actor._sync_joints_from_robot()
    snap = dict(actor.current_joints)
    print(f"  ✓ 记录: {{{', '.join(f'{n}={snap[n]:.2f}' for n in actor.JOINT_NAMES)}}}")
    return snap


# ---------------------------------------------------------------------------
# Keyboard nudge mode  (fallback if no leader arm)
# ---------------------------------------------------------------------------

def _interactive_nudge(actor, prompt: str, step_size: float = 1.0) -> dict:
    """
    Software teleop: keyboard nudges each joint by step_size per keypress.
    Returns recorded joint dict when user hits ENTER.
    """
    import sys, termios, tty, select, copy

    KEYMAP = {
        # key: (joint_name, delta_multiplier)
        'a': ("shoulder_pan",  -1), 'd': ("shoulder_pan",  +1),
        'w': ("shoulder_lift", +1), 's': ("shoulder_lift", -1),  # w=up (less negative)
        'x': ("elbow_flex",    -1), 'c': ("elbow_flex",    +1),
        'q': ("wrist_flex",    -1), 'e': ("wrist_flex",    +1),
        'r': ("wrist_roll",    -1), 'f': ("wrist_roll",    +1),
        't': ("gripper",       -1), 'g': ("gripper",       +1),
        'A': ("shoulder_pan",  -5), 'D': ("shoulder_pan",  +5),  # caps = bigger step
        'W': ("shoulder_lift", +5), 'S': ("shoulder_lift", -5),
    }

    print("\n" + prompt)
    print("─" * 60)
    print("  shoulder_pan:  a / d  (左右扫)        大步: A / D")
    print("  shoulder_lift: s / w  (上下抬)        大步: S / W")
    print("  elbow_flex:    x / c  (肘弯)")
    print("  wrist_flex:    q / e  (腕翻)")
    print("  wrist_roll:    r / f  (腕滚)")
    print("  gripper:       t / g  (夹爪)")
    print("  ENTER 记录   |   ESC 取消")
    print("─" * 60)

    actor._sync_joints_from_robot()
    target = dict(actor.current_joints)

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        last_print = 0.0
        while True:
            # Print current joints periodically (overwrite line)
            now = time.time()
            if now - last_print > 0.2:
                vals = "  ".join(f"{n[:5]}={target[n]:+6.2f}" for n in actor.JOINT_NAMES)
                sys.stdout.write(f"\r {vals}     ")
                sys.stdout.flush()
                last_print = now

            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)

            if ch in ("\r", "\n"):
                # ENTER - record
                actor._sync_joints_from_robot()
                snapshot = dict(actor.current_joints)
                print(f"\n  ✓ 记录: {{{', '.join(f'{n}={snapshot[n]:.2f}' for n in actor.JOINT_NAMES)}}}")
                return snapshot
            if ch == "\x1b":
                raise KeyboardInterrupt("用户取消")

            if ch in KEYMAP:
                joint, mult = KEYMAP[ch]
                target[joint] = target[joint] + mult * step_size
                # Clamp to safety limits
                lo, hi = getattr(actor.limits, joint, (-100, 100))
                target[joint] = max(lo, min(hi, target[joint]))
                # Send small fast move so it tracks key press
                try:
                    actor.go_to_pose(target, duration=0.12, steps=3)
                except Exception as e:
                    logger.warning("go_to_pose failed: %s", e)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)


def _calibrate_robot_with_actor(actor, grid_n: int = 3,
                                 leader: "LeaderArm" = None) -> tuple:
    """
    Per-point calibration. Two modes:
      • leader given  → real teleop, ENTER records
      • leader None   → software keyboard nudge
    """
    use_teleop = leader is not None and leader.connected
    if use_teleop:
        print("\n=== 机械臂工作空间标定 (TELEOP 模式) ===")
        print("用 leader 臂控制 follower 到位，终端按 ENTER 记录。")
    else:
        print("\n=== 机械臂工作空间标定 (KEYBOARD 模式) ===")
        print("用 a/d/w/s/q/e/x/c 等键微调每个关节。")

    pts = grid_points(grid_n)
    N = len(pts)
    joint_names = list(actor.JOINT_NAMES)
    hover = np.zeros((N, len(joint_names)))
    hit   = np.zeros((N, len(joint_names)))

    print(f"\n将采样 {N} 个网格点 × 2 = {2*N} 次记录。")
    print("提示：hover = 物体正上方 3–5cm，hit = 轻触桌面（不要硬压）。")
    print("起点会从上一个记录的位置出发，所以推荐按行移动以减少调整量。\n")

    # Network of grid coords: a 3×3 grid corresponds to A4 paper layout
    #   (0,0)===(0.5,0)===(1,0)         前左 — 前中 — 前右
    #     |       |        |
    #   (0,0.5)=(0.5,0.5)=(1,0.5)       左 — 中央 — 右
    #     |       |        |
    #   (0,1)===(0.5,1)===(1,1)         后左 — 后中 — 后右
    pos_label = {
        (0.0, 0.0): "前左角", (0.5, 0.0): "前中", (1.0, 0.0): "前右角",
        (0.0, 0.5): "中左",   (0.5, 0.5): "正中", (1.0, 0.5): "中右",
        (0.0, 1.0): "后左角", (0.5, 1.0): "后中", (1.0, 1.0): "后右角",
    }

    record = _teleop_record if use_teleop else _interactive_nudge

    try:
        for i, (gx, gy) in enumerate(pts):
            label = pos_label.get((round(gx, 2), round(gy, 2)),
                                  f"({gx:.2f},{gy:.2f})")
            print(f"\n━━━━━━━━━━ 点 {i+1}/{N}  [{label}]  网格 ({gx:.2f},{gy:.2f}) ━━━━━━━━━━")

            snap = record(actor,
                          f">>> 请把夹爪 hover 在 A4 纸的 [{label}] 上方 3-5cm")
            hover[i] = np.array([snap[j] for j in joint_names])

            snap = record(actor,
                          f">>> 现在让夹爪 hit 轻触 [{label}] 桌面位置")
            hit[i] = np.array([snap[j] for j in joint_names])
    except KeyboardInterrupt:
        print("\n⚠️ 用户中止")
        raise

    return pts, hover, hit, joint_names


# ---------------------------------------------------------------------------
# Main
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


def _move_to_observation_pose(actor) -> None:
    """Send arm to the same pose game.py uses for observation."""
    poses = actor.poses
    pose = poses.get("center_look") or poses.get("home")
    if not pose:
        raise RuntimeError("config.yaml 里没有 center_look 也没有 home pose")
    print("→ 移动机械臂到 center_look (这是游戏运行时观察的姿势)...")
    actor.go_to_pose(pose, duration=1.5, steps=30)
    time.sleep(0.5)
    print("✓ 机械臂已就位。")


def _save_pose_to_config(cfg_path: str, pose_name: str, pose_dict: dict) -> bool:
    """
    In-place update of one pose in config.yaml, preserving comments and formatting.
    Uses regex to swap the joint lines under `    pose_name:`.
    """
    import re
    path = Path(cfg_path)
    text = path.read_text(encoding="utf-8")

    joint_block = "\n".join(
        f"      {k}: {round(float(pose_dict[k]), 2)}"
        for k in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]
        if k in pose_dict
    )
    new_block = f"    {pose_name}:\n{joint_block}\n"

    # Match: "    name:\n" + 1+ lines starting with at least 6 spaces
    pattern = re.compile(
        rf"^( {{4}}){re.escape(pose_name)}:[ \t]*\n(?:\1  [^\n]*\n)+",
        re.MULTILINE,
    )
    new_text, n = pattern.subn(new_block, text, count=1)
    if n == 0:
        logger.warning("没在 %s 里找到 pose '%s'，无法原位更新", cfg_path, pose_name)
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def _adjust_observation_pose(actor, leader, camera_cfg: dict, cfg_path: str) -> bool:
    """
    Interactive: user adjusts center_look so all 4 corners of the calibration
    rectangle (e.g. A4 paper) fit comfortably in the camera frame.
    Saves the new pose into config.yaml on SPACE/ENTER. ESC keeps current.
    """
    import cv2
    cap = cv2.VideoCapture(camera_cfg["index"])
    if not cap.isOpened():
        print("⚠️  无法打开摄像头，跳过 center_look 调整。")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])

    using_leader = leader is not None and leader.connected
    win = "Adjust center_look — SPACE save, ESC keep, R reset"
    cv2.namedWindow(win)

    print("\n=== 调整 center_look 观察姿势 ===")
    print("目标：让 A4 纸的 4 个角都清晰显示在画面里（最好留点边距）。")
    if using_leader:
        print("操作：用 leader 臂调整 follower 位置。")
    else:
        print("操作 (键盘): a/d=pan  w/s=lift  c/x=elbow  e/q=wrist  r/f=roll  t/g=gripper")
        print("              大写 = 5° 大步")
    print("看好后按 SPACE 保存为新的 center_look。ESC 保留原值。\n")

    KEYMAP = {
        ord('a'): ("shoulder_pan",  -1), ord('d'): ("shoulder_pan",  +1),
        ord('w'): ("shoulder_lift", +1), ord('s'): ("shoulder_lift", -1),
        ord('x'): ("elbow_flex",    -1), ord('c'): ("elbow_flex",    +1),
        ord('q'): ("wrist_flex",    -1), ord('e'): ("wrist_flex",    +1),
        ord('r'): ("wrist_roll",    -1), ord('f'): ("wrist_roll",    +1),
        ord('t'): ("gripper",       -1), ord('g'): ("gripper",       +1),
        ord('A'): ("shoulder_pan",  -5), ord('D'): ("shoulder_pan",  +5),
        ord('W'): ("shoulder_lift", +5), ord('S'): ("shoulder_lift", -5),
        ord('Z'): ("elbow_flex",    -5), ord('C'): ("elbow_flex",    +5),
    }

    actor._sync_joints_from_robot()
    target = dict(actor.current_joints)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        disp = frame.copy()

        # Reference markers at frame margins
        margin = 25
        for pt in [(margin, margin), (w - margin, margin),
                   (w - margin, h - margin), (margin, h - margin)]:
            cv2.drawMarker(disp, pt, (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        # Center cross
        cv2.line(disp, (w // 2 - 12, h // 2), (w // 2 + 12, h // 2), (255, 255, 255), 1)
        cv2.line(disp, (w // 2, h // 2 - 12), (w // 2, h // 2 + 12), (255, 255, 255), 1)

        # Bottom status bar
        cv2.rectangle(disp, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(disp, "Goal: 4 corners of A4 paper inside the yellow X marks",
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        mode_text = "LEADER mode (move leader arm)" if using_leader \
                    else "KEYBOARD mode (a/d/w/s...)"
        cv2.putText(disp, f"{mode_text}   SPACE save | ESC cancel",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            print("✗ 保留原 center_look (未修改)")
            return False
        if key in (32, 13, 10):  # SPACE / ENTER
            actor._sync_joints_from_robot()
            new_pose = dict(actor.current_joints)
            cap.release()
            cv2.destroyAllWindows()
            ok = _save_pose_to_config(cfg_path, "center_look", new_pose)
            if ok:
                actor.poses["center_look"] = new_pose
                print(f"✓ 新 center_look 已写入 {cfg_path}:")
                for k, v in new_pose.items():
                    print(f"    {k}: {round(v, 2)}")
            else:
                print("⚠️  写入 config.yaml 失败，但内存里已更新（这次会用新值）。")
            return True

        # Keyboard nudge (only if no leader)
        if not using_leader and key in KEYMAP:
            joint, mult = KEYMAP[key]
            target[joint] = target[joint] + mult * 1.0
            lo, hi = getattr(actor.limits, joint, (-100, 100))
            target[joint] = max(lo, min(hi, target[joint]))
            try:
                actor.go_to_pose(target, duration=0.12, steps=3)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--robot-only", action="store_true")
    parser.add_argument("--adjust-center-look", action="store_true",
                        help="只跑 center_look 调整步骤，跳过摄像头/机械臂标定")
    parser.add_argument("--skip-adjust", action="store_true",
                        help="跳过 center_look 调整，直接用 config 里的旧值")
    parser.add_argument("--grid", type=int, default=3, help="N×N grid (default 3)")
    parser.add_argument("--output", default="calibration.npz")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--leader-port", default=None,
                        help="leader 臂串口；省略时读 config.yaml 的 robot.leader_port")
    parser.add_argument("--keyboard", action="store_true",
                        help="强制软件键盘模式（无 leader 时用）")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_path = Path(args.output)
    leader_port = args.leader_port or cfg["robot"].get("leader_port", "")

    # Load existing calibration if partial
    existing = {}
    if out_path.exists() and (args.camera_only or args.robot_only):
        try:
            data = np.load(out_path, allow_pickle=True)
            existing = {k: data[k] for k in data.files}
            print(f"加载已有标定: {out_path}")
        except Exception as e:
            logger.warning("无法加载已有标定: %s", e)

    homography = existing.get("homography")
    grid_xy = existing.get("grid_xy")
    hover = existing.get("hover_joints")
    hit = existing.get("hit_joints")
    joint_names = existing.get("joint_names")
    if joint_names is not None and joint_names.dtype.kind in "OSU":
        joint_names = list(joint_names)

    actor = None
    leader = None
    teleop_thread = None
    teleop_stop = None

    def _try_connect_leader_and_teleop():
        """Connect leader if configured and start teleop loop. Returns (leader, thread, stop)."""
        if not leader_port or args.keyboard:
            return None, None, None
        try:
            print(f"→ 连接 leader 臂 ({leader_port}) ...")
            ldr = LeaderArm(leader_port,
                            robot_id=cfg["robot"].get("leader_id", "zhenbang_leader"),
                            robot_type=cfg["robot"].get("type", "so101"))
            ldr.connect()
            t, stop = _start_teleop_thread(ldr, actor)
            time.sleep(0.5)
            return ldr, t, stop
        except Exception as e:
            logger.warning("Leader 连接/teleop 启动失败: %s — 切到键盘模式", e)
            return None, None, None

    def _stop_teleop(ldr, t, stop):
        if stop: stop.set()
        if t: t.join(timeout=1.0)
        if ldr is not None:
            try: ldr.disconnect()
            except Exception: pass

    # ---- 准备：连 follower ----
    if args.adjust_center_look or not args.camera_only or not args.robot_only:
        actor = _make_actor(cfg)
        actor.connect(calibrate=False)

    # ---- Step 0: 调整 center_look（让摄像头能看到 4 个角）----
    needs_adjust = args.adjust_center_look or (
        not args.skip_adjust and not args.robot_only and not args.camera_only)
    if needs_adjust:
        leader, teleop_thread, teleop_stop = _try_connect_leader_and_teleop()
        try:
            _move_to_observation_pose(actor)
            _adjust_observation_pose(actor, leader, cfg["camera"], args.config)
        finally:
            _stop_teleop(leader, teleop_thread, teleop_stop)
            leader, teleop_thread, teleop_stop = None, None, None

        if args.adjust_center_look:
            # only running the adjust step → skip the rest
            try: actor.disconnect()
            except Exception: pass
            return

    # ---- Step 1: 摄像头 homography ----
    if not args.robot_only:
        try:
            _move_to_observation_pose(actor)  # 重新到（可能更新了的）center_look
            cc = cfg["camera"]
            print("\n机械臂保持不动！现在标定摄像头...")
            homography = calibrate_camera(cc["index"], cc["width"], cc["height"])
        except Exception as e:
            logger.exception("摄像头标定失败: %s", e)
            try: actor.disconnect()
            except Exception: pass
            raise

    # ---- Step 2: 机械臂网格采样 ----
    if not args.camera_only:
        leader, teleop_thread, teleop_stop = _try_connect_leader_and_teleop()
        if leader is None and not args.keyboard:
            if not leader_port:
                print("⚠️ config.yaml 没有 robot.leader_port → 使用键盘模式")
        try:
            grid_xy, hover, hit, joint_names = _calibrate_robot_with_actor(
                actor, grid_n=args.grid, leader=leader)
        finally:
            _stop_teleop(leader, teleop_thread, teleop_stop)

    # ---- 结束: 回 home, 断开 leader/follower ----
    if leader is not None:
        try: leader.disconnect()
        except Exception: pass

    if actor is not None:
        try:
            home = actor.poses.get("home")
            if home:
                print("→ 回到 home 姿势...")
                actor.go_to_pose(home, duration=1.5, steps=30)
            actor.disconnect()
        except Exception:
            pass

    if homography is None or grid_xy is None:
        print("\n⚠️ 标定不完整。再跑一次 --camera-only 或 --robot-only 补齐缺失部分。")

    save_dict = {}
    if homography is not None:
        save_dict["homography"] = homography
    if grid_xy is not None:
        save_dict["grid_xy"] = grid_xy
        save_dict["hover_joints"] = hover
        save_dict["hit_joints"] = hit
        save_dict["joint_names"] = np.array(joint_names)
    np.savez(out_path, **save_dict)
    print(f"\n✓ 标定已保存到 {out_path}")
    print(f"  字段: {list(save_dict.keys())}")


if __name__ == "__main__":
    main()
