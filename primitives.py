"""
Phase I — Closed-loop Primitives Layer (Layer 2 of CaP architecture).

Each primitive is a small, composable, closed-loop unit that the agent layer
(cap_agent.py) can call. Critically, these primitives DO NOT need any new
training — they reuse:
  - detector.py (Phase B Qwen2.5-VL bbox grounding)
  - kinematics.WorkspaceMap (Phase B RBF-interpolated joint angles)
  - robot_actor.py (lerobot smooth-go-to-pose primitive)

Closed-loop strategy:
  - Visual servoing in this codebase uses the calibrated wrist-cam homography,
    which is only valid AT a fixed observation pose (the calibrated pose).
    So we observe → plan world target → move → re-observe → re-plan etc.
    Each "iteration" is observe-then-move, not continuous tracking.
  - Grasp verification: dual-channel — gripper joint angle AFTER closing,
    plus visual lift-and-look check.
  - Retry on grasp failure: micro x-y wiggle + slightly deeper descent.

Returns: every primitive returns (success: bool, info: str).
Void primitives (release, go_home) return (True, "...") on completion.
"""

from __future__ import annotations
import logging
import time
from typing import Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable constants — adjust if behavior is off on your hardware.
# ---------------------------------------------------------------------------

# Gripper joint values (so101 limits: 0..100. 0=fully open, 100=fully closed)
# OPEN: 0.0 = maximally open (was 5.0, but user reported finger spread too narrow
#       and gripper was bumping/pushing carrot during descent).
GRIP_OPEN_VALUE = 0.0  # fully open
GRIP_CLOSE_VALUE = 80.0  # commanded close (will stop earlier if obj inside)

# After commanding close=80, the actual angle reached differs:
#  - empty grip:  ~78.8 on this hardware (measured reliably)
#  - holding thin carrot: ALSO ~78.8 (gripper closes nearly all the way, just
#                                       a hair short because carrot is thin)
# So angle alone is UNRELIABLE for thin objects. We use visual verify
# (lift + re-detect) as ground truth, angle is informational.
# Range bumped to 30..80 to accept "closed but possibly holding".
GRIP_HOLDING_MIN = 30.0  # below = grip didn't even close (command failed)
GRIP_HOLDING_MAX = 80.0  # above = clearly empty (gripper slammed shut)

# Visual servo convergence threshold (in normalized world coords [0,1]²)
WORLD_CONVERGE_THRESH = 0.04

# Visual verify after grasp: if target reappears within this radius
# of the original world position, consider grasp failed (target still on table)
WORLD_REAPPEAR_RADIUS = 0.12

# Per-retry deeper descent (degrees added to shoulder_lift on each retry)
RETRY_DEEPER_DEG = 3.0

# Per-retry wiggle (small offsets in world coords applied at hit pose)
RETRY_WIGGLE = [(0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)]

# Camera-gripper calibration offset (world coords).
# Applied at GRASP time only (not approach) — bbox center is where we LOOK,
# offset is where we actually clamp down so the fingers grip the edge of
# the object, not the center (where elongated objects squirt out).
# - Negative X = shift grasp point LEFT
# - Positive X = shift grasp point RIGHT
# Default: -0.09 — empirically found to work for SO-101 + wrist cam on
# this user's setup (carrot grasped successfully at this offset).
# Tunable at runtime via UI slider.
GRASP_X_OFFSET = -0.09
GRASP_Y_OFFSET = 0.0

# Auto-calibration max offset magnitude (world coords [0,1]²).
# If marker-derived offset exceeds this, it's likely homography extrapolation
# error (marker is near frame edge, far from calibration points). Fall back to
# memory/manual offset instead.
MAX_AUTOCALIB_OFFSET = 0.12


# ---------------------------------------------------------------------------
# CaPPrimitives
# ---------------------------------------------------------------------------


class CaPPrimitives:
    """Layer 2 — closed-loop primitives that the agent layer composes."""

    def __init__(
        self, robot, vision, detector, workspace, observe_pose_name: str = "center_look", dp_runner=None
    ):
        """
        Args
        ----
        robot:    RobotActor
        vision:   VisionSystem
        detector: ObjectDetector (Phase B, supports detect() + detect_target())
        workspace: kinematics.WorkspaceMap (Phase B RBF + homography)
        observe_pose_name: which pose in robot.poses defines where the homography
            is valid. The calibration was taken from this pose.
        dp_runner: Optional DPRunner instance for learned motor control.
        """
        self.robot = robot
        self.vision = vision
        self.detector = detector
        self.workspace = workspace
        self.observe_pose_name = observe_pose_name

        # Telemetry — UI can read these
        self.last_world_target: Optional[Tuple[float, float]] = None
        self.last_grip_angle: Optional[float] = None
        self.last_step_log: list = []  # rolling list of (timestamp, step, info)

        # Camera-gripper offset (UI tunable at runtime). Applied during grasp only.
        self.grasp_x_offset: float = GRASP_X_OFFSET
        self.grasp_y_offset: float = GRASP_Y_OFFSET

        # Persistent grasp memory — remembers what offsets worked per-target.
        try:
            from grasp_memory import GraspMemory

            self.memory = GraspMemory()
        except Exception as e:
            logger.warning("GraspMemory unavailable: %s", e)
            self.memory = None

        # Diffusion Policy runner (learned motor control)
        self.dp_runner = dp_runner

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, step: str, info: str = ""):
        self.last_step_log.append((time.time(), step, info))
        if len(self.last_step_log) > 60:
            self.last_step_log = self.last_step_log[-60:]
        logger.info("[primitive] %s | %s", step, info)

    def _go_pose_with_gripper(
        self,
        joints: Dict[str, float],
        gripper_override: Optional[float] = None,
        duration: float = 1.0,
        steps: int = 20,
    ):
        """Move to joint pose, optionally overriding the gripper command."""
        target = dict(joints)
        if gripper_override is not None:
            target["gripper"] = float(gripper_override)
        self.robot.go_to_pose(target, duration=duration, steps=steps)

    def _detect_gripper_marker(self, frame_bgr=None) -> Optional[Tuple[int, int, int]]:
        """
        Detect black marker on left fingertip in wrist camera frame.
        Returns (cx, cy, area) or None.

        Uses HSV thresholding (S<80, V<60 = black-ish) + position filter
        (must be in bottom half of frame, where gripper appears).
        Empirically tested in test_gripper_marker.py.
        """
        try:
            import cv2
        except ImportError:
            return None
        if frame_bgr is None:
            frame_bgr = self.vision.read_frame() if self.vision else None
        if frame_bgr is None:
            return None
        H, W = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Black mask: low saturation + low value
        mask = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] < 60)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Filter to bottom half (gripper zone) + area range
        candidates = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < 200 or area > 30000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cy = y + h // 2
            if cy / H < 0.4:  # too high, not gripper
                continue
            cx = x + w // 2
            candidates.append((cx, cy, area, cy / H))
        if not candidates:
            return None
        # Pick the one most likely to be the gripper marker (largest area + lowest)
        candidates.sort(key=lambda t: -(t[2] * t[3]))  # area × cy_norm
        cx, cy, area, _ = candidates[0]
        return (cx, cy, area)

    def measure_camera_gripper_offset(self) -> Optional[Tuple[float, float, dict]]:
        """
        Auto-calibrate the camera-gripper offset using the visible marker.

        Returns (offset_x_world, offset_y_world, debug_info) or None if
        marker not visible. The offset is in normalized [0,1]² world coords:
        if camera points at world (wx, wy), gripper is at
        (wx + offset_x, wy + offset_y).

        Caller can use this to replace manual GRASP_X_OFFSET tuning.
        """
        frame = self.vision.read_frame() if self.vision else None
        if frame is None:
            return None
        H, W = frame.shape[:2]
        marker = self._detect_gripper_marker(frame)
        if marker is None:
            return None
        cx, cy, area = marker

        # Project marker pixel and image-center pixel to world coords
        try:
            marker_world = self.workspace.pixel_to_world(cx, cy)
            center_world = self.workspace.pixel_to_world(W // 2, H // 2)
        except Exception as e:
            logger.warning("pixel_to_world failed in auto-calibrate: %s", e)
            return None

        # Offset = gripper_world - camera_pointing_world (= center_world)
        dx = marker_world[0] - center_world[0]
        dy = marker_world[1] - center_world[1]

        debug = {
            "marker_pixel": (cx, cy),
            "marker_area": area,
            "marker_world": marker_world,
            "center_world": center_world,
            "image_size": (W, H),
        }
        return (dx, dy, debug)

    def _carrot_bbox_area_now(self) -> Optional[int]:
        """
        Fast color-based detection of orange carrot. Returns bbox area in pixels²
        (or None if not visible). Used for proximity feedback during descent —
        much faster than Qwen (5ms vs 1000ms).
        """
        try:
            import cv2
        except ImportError:
            return None
        frame = self.vision.read_frame() if self.vision else None
        if frame is None:
            return None
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Orange HSV range (matches detector.py carrot definition)
        lo = np.array([5, 120, 80])
        hi = np.array([22, 255, 255])
        mask = cv2.inRange(hsv, lo, hi)
        # Morph clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        area = int(cv2.contourArea(best))
        return area if area > 200 else None

    def apply_grasp_offset(self, wx: float, wy: float) -> Tuple[float, float]:
        """
        Apply camera-gripper calibration offset.
        Used at grasp time (NOT approach) so the gripper closes at the
        carrot's edge instead of dead center (elongated objects squirt out
        from center-clamp).
        """
        return (wx + self.grasp_x_offset, wy + self.grasp_y_offset)

    def _go_pose_preserve_grip(self, joints: Dict[str, float], duration: float = 1.0, steps: int = 20):
        """
        Move to joint pose KEEPING the current gripper value. Use this any
        time we may be holding an object — RBF interpolated poses include
        a default gripper value that would open the gripper and drop the
        object mid-motion.
        """
        self.robot._sync_joints_from_robot()
        current_grip = float(self.robot.current_joints.get("gripper", 0.0))
        target = dict(joints)
        target["gripper"] = current_grip
        self.robot.go_to_pose(target, duration=duration, steps=steps)

    # Target name aliases — bridges English LLM output ↔ Chinese detector labels.
    # Used when free-text VLM detect_target unavailable (Qwen down / API key missing)
    # and we fall back to the color / YOLO backend which returns Chinese names.
    TARGET_ALIASES = {
        "carrot": {"carrot", "萝卜", "🥕 萝卜", "🥕萝卜", "胡萝卜"},
        "tissue": {"tissue", "tissue roll", "纸巾", "🧻 纸巾", "🧻纸巾"},
        "cola": {"cola", "cola can", "可乐", "🥤 可乐", "🥤可乐"},
        "萝卜": {"carrot", "萝卜", "🥕 萝卜", "🥕萝卜", "胡萝卜"},
        "纸巾": {"tissue", "tissue roll", "纸巾", "🧻 纸巾", "🧻纸巾"},
        "可乐": {"cola", "cola can", "可乐", "🥤 可乐", "🥤可乐"},
    }

    @classmethod
    def _name_matches(cls, target_name: str, detected_name: str) -> bool:
        """True if detected_name matches target_name (or a known alias)."""
        if target_name == detected_name:
            return True
        if target_name in detected_name or detected_name in target_name:
            return True
        aliases = cls.TARGET_ALIASES.get(target_name, set())
        if detected_name in aliases:
            return True
        # also check if any alias is substring of detected name (handles emojis)
        for alias in aliases:
            if alias in detected_name or detected_name in alias:
                return True
        return False

    def _detect_target_pixel(self, target_name: str) -> Optional[Tuple[int, int, dict]]:
        """
        Return (cx, cy, det_dict) or None.
        Tries free-text detect_target first (works for any string),
        falls back to multi-class detect() if free-text returns nothing.
        Cross-language name matching via TARGET_ALIASES.
        """
        frame = self.vision.read_frame() if self.vision else None
        if frame is None:
            return None

        # Try free-text query first (needs VLM backend)
        try:
            det = self.detector.detect_target(frame, target_name)
        except Exception as e:
            logger.warning("detect_target raised: %s", e)
            det = None

        if det is None:
            # Fall back: multi-class detect (works on color/yolo backends too)
            try:
                dets = self.detector.detect(frame, top_k_per_class=1)
            except Exception as e:
                logger.warning("detect raised: %s", e)
                dets = []
            for d in dets:
                if self._name_matches(target_name, d.get("name", "")):
                    det = d
                    break

        if det is None:
            return None

        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return (int(cx), int(cy), det)

    def _pixel_to_world_safe(self, cx: int, cy: int) -> Optional[Tuple[float, float]]:
        """Apply homography, return None if out of calibrated bounds."""
        try:
            wx, wy = self.workspace.pixel_to_world(cx, cy)
        except Exception as e:
            logger.warning("pixel_to_world failed: %s", e)
            return None
        if not self.workspace.in_bounds(wx, wy, margin=0.1):
            self._log("warn", f"target outside workspace bounds: ({wx:.2f},{wy:.2f})")
            return None
        return (float(wx), float(wy))

    # ------------------------------------------------------------------
    # PUBLIC PRIMITIVES — the agent calls these
    # ------------------------------------------------------------------

    def go_to_observe(self) -> Tuple[bool, str]:
        """
        Move to the calibrated observation pose. Homography is only valid here.
        Gripper is opened so the wrist cam isn't blocked by closed fingers.
        """
        pose = self.robot.poses.get(self.observe_pose_name)
        if pose is None:
            pose = self.robot.poses.get("home")
        if pose is None:
            return False, "no observe pose"
        self._go_pose_with_gripper(pose, gripper_override=GRIP_OPEN_VALUE, duration=0.9, steps=18)
        time.sleep(0.3)
        self._log("go_to_observe", "ok")
        return True, "at observe pose"

    def locate_target(self, target_name: str = "carrot") -> Optional[Tuple[float, float]]:
        """Observe → detect → world coords. Returns (wx, wy) or None."""
        ok, _ = self.go_to_observe()
        if not ok:
            return None
        result = self._detect_target_pixel(target_name)
        if result is None:
            self._log("locate_target", f"not_found:{target_name}")
            return None
        cx, cy, _ = result
        world = self._pixel_to_world_safe(cx, cy)
        if world is None:
            return None
        self.last_world_target = world
        self._log("locate_target", f"{target_name} px=({cx},{cy}) world=({world[0]:.2f},{world[1]:.2f})")
        return world

    def approach(self, target_name: str = "carrot", max_iters: int = 3) -> Tuple[bool, str]:
        """
        Closed-loop approach: observe → plan world target → move to hover
        above it → re-observe to verify → repeat until convergence or max_iters.

        Convergence: world-coord change between iterations < WORLD_CONVERGE_THRESH.
        """
        last_target = None
        for i in range(max_iters):
            world = self.locate_target(target_name)
            if world is None:
                return False, f"target_lost (iter {i})"

            # Move to hover above world target
            try:
                hover_joints = self.workspace.joints_at(world[0], world[1], mode="hover")
            except Exception as e:
                return False, f"joints_at failed: {e}"

            self._go_pose_with_gripper(hover_joints, gripper_override=GRIP_OPEN_VALUE, duration=0.8, steps=16)
            time.sleep(0.4)
            self._log("approach", f"iter {i} hover at ({world[0]:.2f},{world[1]:.2f})")

            # Convergence check (after at least one move)
            if last_target is not None:
                dx = abs(world[0] - last_target[0])
                dy = abs(world[1] - last_target[1])
                if dx < WORLD_CONVERGE_THRESH and dy < WORLD_CONVERGE_THRESH:
                    return True, f"converged after {i + 1} iters"

            last_target = world

        # Out of iterations but still in valid state
        return True, f"max_iters reached, last={last_target}"

    def _grasp_with_dp(
        self, target_name: str, world_xy: Tuple[float, float] | None = None
    ) -> Tuple[bool, str] | None:
        """
        Try grasping using Diffusion Policy learned motor control.

        Flow: locate → hover → DP closed-loop (descend+grasp+lift) → verify.
        If world_xy is provided (e.g. from CodeGen), skip redundant observe+detect.

        Returns (True, ...) on success, (False, ...) on failure,
        None if DP is not suitable (e.g. vision not ready).
        """
        if self.vision is None:
            return None

        if world_xy is not None:
            world = world_xy
        else:
            world = self.locate_target(target_name)
            if world is None:
                return False, f"DP: target not found ({target_name})"

        wx, wy = world
        wx += self.grasp_x_offset
        wy += self.grasp_y_offset

        try:
            hover_joints = self.workspace.joints_at(wx, wy, mode="hover")
        except Exception as e:
            return False, f"DP: joints_at failed: {e}"
        self._go_pose_with_gripper(hover_joints, gripper_override=GRIP_OPEN_VALUE, duration=0.9, steps=18)
        time.sleep(0.3)

        self._log("grasp_with_dp", f"hover at ({wx:.2f},{wy:.2f}), starting DP loop")

        self.dp_runner.reset()
        DP_DURATION = 10.0
        DP_HZ = 20.0
        max_steps = int(DP_DURATION * DP_HZ)
        dt = 1.0 / DP_HZ
        dp_progress = []

        for step in range(max_steps):
            t_step = time.time()
            frame = self.vision.read_frame() if self.vision else None
            if frame is None:
                time.sleep(dt)
                continue

            self.robot._sync_joints_from_robot()
            state = dict(self.robot.current_joints)
            grip_before = float(state.get("gripper", 0.0))

            action = self.dp_runner.select_action(frame, state)
            if action is not None:
                self.robot.send_action(action)

            if step % 5 == 0:
                self.robot._sync_joints_from_robot()
                grip_now = float(self.robot.current_joints.get("gripper", 0.0))
                ag = action.get("gripper.pos", 0) if action else 0
                info = f"step={step}/{max_steps} grip={grip_before:.0f}→{grip_now:.0f} cmd_grip={ag:.0f}"
                dp_progress.append(f"{time.strftime('%H:%M:%S', time.localtime(t_step))} {info}")
                self._log("dp_loop", info)

            elapsed = time.time() - t_step
            if elapsed < dt:
                time.sleep(dt - elapsed)

        self.robot._sync_joints_from_robot()
        grip_angle = float(self.robot.current_joints.get("gripper", 0.0))
        self.last_grip_angle = grip_angle
        self._log("grasp_with_dp", f"DP done, grip_angle={grip_angle:.1f}")

        try:
            hover_joints = self.workspace.joints_at(wx, wy, mode="hover")
        except Exception as e:
            return False, f"DP: lift joints_at failed: {e}"
        self._go_pose_preserve_grip(hover_joints, duration=0.8, steps=16)
        time.sleep(0.3)

        visual_ok = self._verify_grasp_visual(target_name, world)
        if visual_ok:
            if self.memory:
                try:
                    self.memory.record_success(target_name, self.grasp_x_offset, self.grasp_y_offset)
                except Exception as e:
                    logger.warning("memory.record_success failed: %s", e)
            return True, f"DP grasp succeeded (grip={grip_angle:.1f}, visual confirmed)"

        self._log("grasp_with_dp", "DP grasp failed (visual verify showed target still on table)")
        self._set_gripper(GRIP_OPEN_VALUE)
        return False, "DP grasp failed, falling back to hand-coded"

    def grasp_with_retry(self, target_name: str = "carrot", max_attempts: int = 3) -> Tuple[bool, str]:
        """
        Re-locate target → descend → close gripper → verify (angle + visual).
        On failure: open, micro-wiggle, descend deeper, retry.

        Memory: if grasp_memory has a recorded offset for this target, use it
        (overrides default GRASP_X/Y_OFFSET for this run). On success, save the
        working offset back to memory.
        """
        # Load offset from memory if available (overrides default for this run)
        if self.memory:
            mem_offset = self.memory.get_offset(target_name)
            if mem_offset is not None:
                old_x, old_y = self.grasp_x_offset, self.grasp_y_offset
                self.grasp_x_offset, self.grasp_y_offset = mem_offset
                self._log(
                    "grasp_attempt",
                    f"loaded offset from memory: ({mem_offset[0]:+.3f},{mem_offset[1]:+.3f}) "
                    f"(was default ({old_x:+.3f},{old_y:+.3f}))",
                )

        # ── DP path: if Diffusion Policy is available, try it first ──
        if self.dp_runner is not None:
            # CodeGen already located target and passed it via last_world
            result = self._grasp_with_dp(target_name)
            if result is not None:
                return result
            # DP failed or not applicable, fall through to hand-coded

        last_world: Optional[Tuple[float, float]] = None

        for attempt in range(max_attempts):
            # Re-locate target each retry (this also moves arm to observe pose,
            # which is required for accurate homography projection).
            world = self.locate_target(target_name)
            if world is None:
                if last_world is None:
                    return False, f"target_lost on attempt {attempt}"
                world = last_world
            last_world = world

            # Auto-calibration via gripper marker (now at observe pose).
            # Falls back to memory/manual if marker not visible or offset unreliable.
            marker_offset = None
            try:
                calib = self.measure_camera_gripper_offset()
                if calib is not None:
                    dx, dy, dbg = calib
                    marker_offset_raw = (-dx, -dy)
                    # Sanity check: reject if offset magnitude exceeds max allowed.
                    # The homography is only reliable near calibration points; the
                    # marker at the frame edge produces extreme extrapolation errors.
                    mag = float(np.hypot(marker_offset_raw[0], marker_offset_raw[1]))
                    if mag > MAX_AUTOCALIB_OFFSET:
                        self._log(
                            "grasp_attempt",
                            f"auto-calib offset mag={mag:.3f} > {MAX_AUTOCALIB_OFFSET} "
                            f"(marker_px={dbg['marker_pixel']} near frame edge), "
                            f"rejecting — using memory offset ({self.grasp_x_offset:+.3f},{self.grasp_y_offset:+.3f})",
                        )
                    else:
                        marker_offset = marker_offset_raw
                        self._log(
                            "grasp_attempt",
                            f"auto-calib at observe: marker_px={dbg['marker_pixel']} "
                            f"→ offset=({marker_offset[0]:+.3f},{marker_offset[1]:+.3f}) "
                            f"vs memory ({self.grasp_x_offset:+.3f},{self.grasp_y_offset:+.3f}) "
                            f"Δ=({marker_offset[0] - self.grasp_x_offset:+.3f},{marker_offset[1] - self.grasp_y_offset:+.3f})",
                        )
                else:
                    self._log("grasp_attempt", "auto-calib: marker not visible (using memory)")
            except Exception as e:
                self._log("grasp_attempt", f"auto-calib error: {e} (using memory)")

            wx, wy = world
            # Use marker-derived offset if available AND passed sanity check,
            # else fall back to memory/manual offset (empirically validated).
            if marker_offset is not None:
                ox, oy = marker_offset
            else:
                ox, oy = self.grasp_x_offset, self.grasp_y_offset
            wx, wy = wx + ox, wy + oy
            # Apply per-attempt wiggle on top
            wig_dx, wig_dy = RETRY_WIGGLE[attempt % len(RETRY_WIGGLE)]
            wx_try = wx + wig_dx
            wy_try = wy + wig_dy
            offset_src = "marker" if marker_offset else "memory"
            self._log(
                "grasp_attempt",
                f"target world=({world[0]:.3f},{world[1]:.3f}) "
                f"+offset[{offset_src}]=({ox:+.3f},{oy:+.3f}) "
                f"+wiggle=({wig_dx:+.3f},{wig_dy:+.3f}) "
                f"→ grasp at ({wx_try:.3f},{wy_try:.3f})",
            )

            # ───────── Phase 1: Open gripper FULLY at observe pose ─────────
            # Hardware is slow — going 80 → 0 takes ~1s. Give plenty of time
            # + wait + verify, retry if hardware didn't actually open.
            self._set_gripper(GRIP_OPEN_VALUE)
            time.sleep(0.8)
            self.robot._sync_joints_from_robot()
            actual_grip = float(self.robot.current_joints.get("gripper", -1))

            if actual_grip > 30:
                # First attempt didn't fully open. Try again with even more time.
                self._log(
                    "grasp_attempt",
                    f"gripper open lag: actual={actual_grip:.1f} > 30, giving it more time...",
                )
                self._set_gripper(GRIP_OPEN_VALUE)
                time.sleep(1.0)
                self.robot._sync_joints_from_robot()
                actual_grip = float(self.robot.current_joints.get("gripper", -1))

            self._log(
                "grasp_attempt",
                f"gripper opened: commanded={GRIP_OPEN_VALUE}, actual={actual_grip:.1f}"
                f"{' ✓ wide' if actual_grip < 15 else ' ⚠ still narrow'}",
            )

            # Compute target joints (hover above + hit at)
            try:
                hover_joints = self.workspace.joints_at(wx_try, wy_try, mode="hover")
                hit_joints = self.workspace.joints_at(wx_try, wy_try, mode="hit")
            except Exception as e:
                return False, f"workspace.joints_at failed: {e}"

            # Per-retry deeper descent — push shoulder_lift further down at hit
            hit_joints = dict(hit_joints)
            if attempt > 0:
                hit_joints["shoulder_lift"] = (
                    hit_joints.get("shoulder_lift", 0.0) + RETRY_DEEPER_DEG * attempt
                )

            # ───────── Phase 2: Move LATERALLY to hover above target ─────────
            # Gripper stays OPEN. This is lateral+up motion, can't bump carrot.
            self._go_pose_with_gripper(hover_joints, gripper_override=GRIP_OPEN_VALUE, duration=0.9, steps=18)
            time.sleep(0.3)

            # Snapshot bbox area at HOVER (baseline for diagnostics — see ratio at hit)
            bbox_at_hover = self._carrot_bbox_area_now()
            self._log(
                "grasp_attempt",
                f"at hover. carrot bbox area = {bbox_at_hover} px²"
                f"{' (baseline)' if bbox_at_hover else ' (not visible)'}",
            )

            # ───────── Phase 3: Descend to hit pose (gripper open) ─────────
            # Quick descent, gripper open. Close happens in Phase 4 at the carrot.
            self._go_pose_with_gripper(hit_joints, gripper_override=GRIP_OPEN_VALUE, duration=1.6, steps=32)
            time.sleep(0.3)

            # Bbox check AFTER descent (informational — see how much closer we got)
            bbox_at_hit = self._carrot_bbox_area_now()
            if bbox_at_hover and bbox_at_hit:
                ratio = bbox_at_hit / bbox_at_hover
                self._log(
                    "grasp_attempt",
                    f"at hit pose. carrot bbox = {bbox_at_hit} px² (grew {ratio:.2f}× since hover)",
                )

            # Read actual gripper angle from hardware
            self.robot._sync_joints_from_robot()
            grip_angle = float(self.robot.current_joints.get("gripper", 0.0))
            self._log(
                "grasp_attempt",
                f"after descent+grip: grip_angle={grip_angle:.1f} "
                f"{'✓ closed' if grip_angle >= GRIP_HOLDING_MIN else 'still open'}",
            )

            # ───────── Phase 4: Close gripper + active wait ─────────
            # Command close, then wait until hardware confirms.
            self._set_gripper(GRIP_CLOSE_VALUE)
            self._log("grasp_attempt", "command close, waiting for hardware...")
            close_start = time.time()
            while time.time() - close_start < 3.0:
                time.sleep(0.1)
                self.robot._sync_joints_from_robot()
                grip_angle = float(self.robot.current_joints.get("gripper", 0.0))
                if grip_angle >= GRIP_HOLDING_MIN:
                    break
            elapsed = time.time() - close_start
            self._log(
                "grasp_attempt",
                f"close: grip_angle={grip_angle:.1f} after {elapsed:.1f}s"
                f"{' ✓' if grip_angle >= GRIP_HOLDING_MIN else ' ✗ still low'}",
            )
            self.last_grip_angle = grip_angle

            # Did the gripper fail to close at all?
            if grip_angle < GRIP_HOLDING_MIN:
                self._log(
                    "grasp_attempt",
                    f"attempt={attempt + 1} grip_angle={grip_angle:.1f} < {GRIP_HOLDING_MIN} → retry",
                )
                self._set_gripper(GRIP_OPEN_VALUE)
                time.sleep(0.2)
                continue

            # ───────── Phase 5: Lift + visual verify (AUTHORITATIVE) ─────────
            # Lift to hover BEFORE judging success. Preserve gripper so we don't
            # drop the object during lift.
            hover_joints = self.workspace.joints_at(wx_try, wy_try, mode="hover")
            self._go_pose_preserve_grip(hover_joints, duration=0.8, steps=16)
            time.sleep(0.3)

            # Visual verify: is the carrot still on the table at the original
            # position? If yes → empty grip. If no → we have it.
            visual_ok = self._verify_grasp_visual(target_name, world)
            self._log(
                "grasp_attempt",
                f"attempt={attempt + 1} grip_angle={grip_angle:.1f} "
                f"visual_verify={'CARROT GONE → HELD' if visual_ok else 'STILL ON TABLE → EMPTY'}",
            )

            if visual_ok:
                # Save the working offset to memory for next time
                if self.memory:
                    try:
                        self.memory.record_success(target_name, self.grasp_x_offset, self.grasp_y_offset)
                    except Exception as e:
                        logger.warning("memory.record_success failed: %s", e)
                return True, f"grasped (attempt {attempt + 1}, grip={grip_angle:.1f}, visual confirmed)"

            # Visual says empty — open + retry
            self._set_gripper(GRIP_OPEN_VALUE)
            time.sleep(0.2)

        # All attempts failed — record in memory
        if self.memory:
            try:
                self.memory.record_failure(target_name)
            except Exception:
                pass
        return False, f"all {max_attempts} attempts failed"

    def _verify_grasp_visual(self, target_name: str, original_world: Tuple[float, float]) -> bool:
        """
        After gripping + lifting, check if target is on the table.

        Logic (FIXED — was previously buggy):
          - target visible on table (anywhere) → STILL THERE → empty grip
          - target NOT visible from observe pose → gone (in gripper or occluded) → HELD
          - target was at original position but now moved sideways → STILL ON TABLE
            (was previously misreported as "moved = HELD". WRONG. Gripper bumps
            carrot to new table position; that's a failed grasp, not a success.)

        Returns True if grasp likely successful (target gone from view).
        """
        observe_pose = self.robot.poses.get(self.observe_pose_name)
        if observe_pose is None:
            observe_pose = self.robot.poses.get("home")
        if observe_pose is None:
            return True  # can't verify, give benefit of doubt

        # Preserve current gripper (we may be holding the object)
        self._go_pose_preserve_grip(observe_pose, duration=0.9, steps=18)
        time.sleep(0.3)

        result = self._detect_target_pixel(target_name)
        if result is None:
            self._log("verify_visual", "target NOT visible → likely held in gripper (or occluded)")
            return True

        # Target IS still visible — that means it's on the table (in gripper would
        # mean it's not detectable in the workspace area from observe pose).
        cx, cy, _ = result
        world = self._pixel_to_world_safe(cx, cy)
        if world is None:
            # Detected in pixel but couldn't project — give benefit of doubt
            self._log(
                "verify_visual", "target seen but out of workspace projection → ambiguous, assuming held"
            )
            return True

        dx = world[0] - original_world[0]
        dy = world[1] - original_world[1]
        d = float(np.hypot(dx, dy))
        self._log(
            "verify_visual",
            f"target STILL ON TABLE at ({world[0]:.2f},{world[1]:.2f}) "
            f"(orig=({original_world[0]:.2f},{original_world[1]:.2f}), "
            f"moved {d:.3f}) → EMPTY GRIP",
        )
        return False

    def transport_to(self, dest_world_xy: Tuple[float, float]) -> Tuple[bool, str]:
        """Move (presumably gripping) to a world (x, y) at hover height."""
        try:
            wx, wy = float(dest_world_xy[0]), float(dest_world_xy[1])
        except (TypeError, IndexError, ValueError):
            return False, f"bad dest: {dest_world_xy}"
        if not self.workspace.in_bounds(wx, wy, margin=0.1):
            return False, f"dest out of workspace: ({wx:.2f},{wy:.2f})"

        try:
            hover = self.workspace.joints_at(wx, wy, mode="hover")
        except Exception as e:
            return False, f"joints_at failed: {e}"

        # Preserve gripper state — RBF hover pose has default gripper value
        # that would drop the object mid-transport
        self._go_pose_preserve_grip(hover, duration=1.4, steps=28)
        time.sleep(0.3)
        self._log("transport_to", f"({wx:.2f},{wy:.2f})")
        return True, f"transported to ({wx:.2f},{wy:.2f})"

    def release(self) -> Tuple[bool, str]:
        """Open gripper to drop object."""
        self._set_gripper(GRIP_OPEN_VALUE)
        time.sleep(0.3)
        self._log("release", "gripper opened")
        return True, "released"

    def open_gripper(self) -> Tuple[bool, str]:
        self._set_gripper(GRIP_OPEN_VALUE)
        return True, "opened"

    def close_gripper(self) -> Tuple[bool, str]:
        self._set_gripper(GRIP_CLOSE_VALUE)
        return True, "closed"

    def go_home(self) -> Tuple[bool, str]:
        home = self.robot.poses.get("home")
        if home is None:
            return False, "no home pose"
        self.robot.go_to_pose(home, duration=1.2, steps=24)
        self._log("go_home", "ok")
        return True, "at home"

    # ------------------------------------------------------------------
    def _set_gripper(self, value: float):
        """Send a gripper-only command, holding other joints at current position."""
        self.robot._sync_joints_from_robot()
        cur = dict(self.robot.current_joints)
        cur["gripper"] = float(value)
        self.robot.go_to_pose(cur, duration=0.8, steps=8)
