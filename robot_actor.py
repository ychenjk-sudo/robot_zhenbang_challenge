"""
Robot Actor Module
Handles LeRobot arm control with smooth interpolation and "cat acting" poses.
"""

import time
import logging
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JointLimits:
    """Safety limits for each joint in degrees or normalized units."""

    shoulder_pan: tuple = (-100, 100)
    shoulder_lift: tuple = (-100, 100)
    elbow_flex: tuple = (-100, 100)
    wrist_flex: tuple = (-100, 100)
    wrist_roll: tuple = (-100, 100)
    gripper: tuple = (0, 100)


class RobotActor:
    """
    Controls a LeRobot arm with smooth motions and predefined poses.
    Acts like the cat 'Dakaimen' from the viral video.
    """

    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    def __init__(
        self,
        robot_type: str = "so100",
        port: str = "/dev/ttyUSB0",
        robot_id: str = "robot",
        smooth_factor: float = 0.12,
        step_delay: float = 0.05,
        poses: Optional[Dict] = None,
    ):
        self.robot_type = robot_type
        self.port = port
        self.robot_id = robot_id
        self.smooth_factor = smooth_factor
        self.step_delay = step_delay
        self.poses = poses or {}
        self.limits = JointLimits()

        self.robot = None
        self.current_joints: Dict[str, float] = {j: 0.0 for j in self.JOINT_NAMES}
        self.connected = False
        self._stop_flag = threading.Event()
        self._motion_lock = threading.Lock()

    def _try_import_robot(self):
        """Instantiate the LeRobot follower class for the configured arm type."""
        if self.robot_type in ("so100", "so101", "so_arm100"):
            from lerobot.robots.so_follower import SOFollower
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

            cfg = SOFollowerRobotConfig(port=self.port, id=self.robot_id)
            return SOFollower(cfg)
        if self.robot_type in ("koch", "koch_follower"):
            from lerobot.robots.koch_follower import KochFollower
            from lerobot.robots.koch_follower.config_koch_follower import KochFollowerConfig

            cfg = KochFollowerConfig(port=self.port, id=self.robot_id)
            return KochFollower(cfg)
        raise ImportError(f"Unsupported robot type: '{self.robot_type}'")

    def connect(self, calibrate: bool = False):
        """Connect to the robot hardware."""
        if self.connected:
            return
        logger.info(f"Connecting to {self.robot_type} on {self.port} ...")
        self.robot = self._try_import_robot()

        # Try context-like connect
        if hasattr(self.robot, "connect"):
            self.robot.connect(calibrate=calibrate)

        # Read current state
        self._sync_joints_from_robot()
        self.connected = True
        logger.info("Robot connected. Current joints: %s", self.current_joints)

    def disconnect(self):
        """Safely disconnect."""
        if self.robot and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
        self.connected = False
        logger.info("Robot disconnected.")

    def _sync_joints_from_robot(self):
        """Read current joint positions from robot observation."""
        if not self.robot:
            return
        try:
            obs = self.robot.get_observation()
            if obs is None:
                return
            # Observation format varies; try common keys
            for j in self.JOINT_NAMES:
                key_candidates = [
                    f"{j}.pos",
                    f"{j}",
                    j,
                    f"observation.{j}.pos",
                ]
                for k in key_candidates:
                    if k in obs:
                        self.current_joints[j] = float(obs[k])
                        break
        except Exception as e:
            logger.warning("Failed to read robot observation: %s", e)

    def _clamp(self, name: str, value: float) -> float:
        """Clamp joint value to safety limits."""
        lo, hi = getattr(self.limits, name, (-100, 100))
        return float(np.clip(value, lo, hi))

    def _build_action_dict(self, joints: Dict[str, float]) -> Dict:
        """Build action dictionary expected by LeRobot."""
        action = {}
        for j in self.JOINT_NAMES:
            val = self._clamp(j, joints.get(j, self.current_joints.get(j, 0.0)))
            # LeRobot expects keys like "shoulder_pan.pos"
            action[f"{j}.pos"] = val
        return action

    def send_action(self, action: Dict[str, float]):
        """
        Send a single action to the robot — fire-and-forget, no interpolation.

        Accepts keys with ".pos" suffix (e.g. ``shoulder_pan.pos``) or plain
        joint names (e.g. ``shoulder_pan``).  The underlying Feetech servo
        will move to the target at its maximum speed.

        Use this from DP closed-loop control where you want 1:1 mapping
        between model output and robot command.
        """
        if not self.connected or self.robot is None:
            logger.warning("Robot not connected, skipping action.")
            return

        # Normalise keys: accept both {"shoulder_pan": v} and {"shoulder_pan.pos": v}
        action_dict = {}
        for j in self.JOINT_NAMES:
            if f"{j}.pos" in action:
                action_dict[f"{j}.pos"] = self._clamp(j, float(action[f"{j}.pos"]))
            elif j in action:
                action_dict[f"{j}.pos"] = self._clamp(j, float(action[j]))
            else:
                action_dict[f"{j}.pos"] = float(self.current_joints.get(j, 0.0))

        try:
            self.robot.send_action(action_dict)
            # Update software state to match target (actual hardware reading is
            # independent via _sync_joints_from_robot).
            for j in self.JOINT_NAMES:
                self.current_joints[j] = float(action_dict[f"{j}.pos"])
        except Exception as e:
            logger.warning("send_action failed: %s", e)

    def go_to_pose(self, pose: Dict[str, float], duration: float = 1.0, steps: int = 30):
        """
        Smoothly move to a target pose using linear interpolation.

        Args:
            pose: Target joint dictionary.
            duration: Total motion time in seconds.
            steps: Number of interpolation steps.
        """
        if not self.connected or self.robot is None:
            logger.warning("Robot not connected, skipping motion.")
            return

        target = {j: float(pose.get(j, self.current_joints[j])) for j in self.JOINT_NAMES}
        start = dict(self.current_joints)

        step_time = duration / max(steps, 1)

        with self._motion_lock:
            for i in range(1, steps + 1):
                if self._stop_flag.is_set():
                    logger.info("Motion interrupted by stop flag.")
                    return

                alpha = i / steps
                # Optional ease-in-out
                alpha = alpha * alpha * (3 - 2 * alpha)

                interp = {j: start[j] + (target[j] - start[j]) * alpha for j in self.JOINT_NAMES}

                action = self._build_action_dict(interp)
                try:
                    self.robot.send_action(action)
                except Exception as e:
                    logger.error("send_action error: %s", e)
                    return

                self.current_joints = dict(interp)
                time.sleep(step_time)

    def quick_tap(self, side: str = "left"):
        """
        Quick tapping motion: hover -> hit -> hover.
        Mimics the cat slapping the item.
        """
        hover = f"{side}_hover"
        hit = f"{side}_hit"

        if hover not in self.poses or hit not in self.poses:
            logger.warning("Missing poses for tap: %s / %s", hover, hit)
            return

        # Move to hover quickly
        self.go_to_pose(self.poses[hover], duration=0.6, steps=15)
        time.sleep(0.2)

        # Downward tap (fast)
        self.go_to_pose(self.poses[hit], duration=0.25, steps=8)
        time.sleep(0.15)

        # Back up
        self.go_to_pose(self.poses[hover], duration=0.25, steps=8)
        time.sleep(0.2)

    def tap_at_world(self, x: float, y: float, workspace=None):
        """
        Phase B: tap at arbitrary (x, y) in calibrated table coords [0,1]².
        Uses RBF-interpolated joint angles from kinematics.WorkspaceMap.

        Falls back to quick_tap with the closest discrete side if no workspace.
        """
        if workspace is None:
            # Fallback: pick discrete zone
            if x < 0.33:
                self.quick_tap("left")
            elif x > 0.67:
                self.quick_tap("right")
            else:
                self.quick_tap("center")
            return

        if not workspace.in_bounds(x, y):
            logger.warning("tap_at_world: (%.2f, %.2f) outside calibrated workspace", x, y)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

        hover_pose = workspace.joints_at(x, y, mode="hover")
        hit_pose = workspace.joints_at(x, y, mode="hit")

        # hover -> hit -> hover
        self.go_to_pose(hover_pose, duration=0.7, steps=18)
        time.sleep(0.2)
        self.go_to_pose(hit_pose, duration=0.25, steps=8)
        time.sleep(0.15)
        self.go_to_pose(hover_pose, duration=0.25, steps=8)
        time.sleep(0.2)

    def hesitate(self, cycles: int = 3):
        """
        Cat-like hesitation: look left, look right, center, repeat.
        """
        left = self.poses.get("left_peek")
        right = self.poses.get("right_peek")
        center = self.poses.get("center_look", self.poses.get("home"))

        if not left or not right:
            logger.warning("Missing peek poses for hesitation.")
            return

        for _ in range(cycles):
            if self._stop_flag.is_set():
                return
            self.go_to_pose(left, duration=0.5, steps=10)
            time.sleep(0.3)
            self.go_to_pose(center, duration=0.4, steps=8)
            time.sleep(0.2)
            self.go_to_pose(right, duration=0.5, steps=10)
            time.sleep(0.3)
            self.go_to_pose(center, duration=0.4, steps=8)
            time.sleep(0.2)

    def tentative_look(self, side: str = "left"):
        """
        Fake-out: move toward one side as if choosing it, then pull back.
        Mimics the cat tapping the wrong item first, checking owner's face.
        """
        peek = self.poses.get(f"{side}_peek")
        center = self.poses.get("center_look", self.poses.get("home"))

        if not peek:
            return

        # Slowly approach
        self.go_to_pose(peek, duration=0.8, steps=20)
        time.sleep(0.6)  # "Checking owner's reaction"

        # Pull back quickly (uh oh wrong?)
        self.go_to_pose(center, duration=0.5, steps=12)
        time.sleep(0.3)

    def celebrate(self):
        """Happy celebration motion + gripper open/close."""
        celebrate_pose = self.poses.get("celebrate")
        home = self.poses.get("home")

        if celebrate_pose:
            self.go_to_pose(celebrate_pose, duration=0.8, steps=16)
            time.sleep(0.3)
            # Wiggle gripper
            wiggle_open = dict(celebrate_pose)
            wiggle_open["gripper"] = 0.0
            wiggle_close = dict(celebrate_pose)
            wiggle_close["gripper"] = 100.0

            for _ in range(3):
                if self._stop_flag.is_set():
                    break
                self.go_to_pose(wiggle_open, duration=0.2, steps=4)
                self.go_to_pose(wiggle_close, duration=0.2, steps=4)
            time.sleep(0.3)

        if home:
            self.go_to_pose(home, duration=1.0, steps=20)

    def sad_reaction(self):
        """Deflated motion when wrong."""
        sad = self.poses.get("sad")
        home = self.poses.get("home")

        if sad:
            self.go_to_pose(sad, duration=1.0, steps=20)
            time.sleep(1.0)

        if home:
            self.go_to_pose(home, duration=1.2, steps=24)

    def stop(self):
        """Signal to stop current motion."""
        self._stop_flag.set()

    def reset_stop(self):
        """Clear stop flag for next motion."""
        self._stop_flag.clear()

    def emergency_home(self):
        """Immediately return to home pose (blocking)."""
        home = self.poses.get("home")
        if home and self.connected:
            self._stop_flag.clear()
            self.go_to_pose(home, duration=1.5, steps=30)
