"""
Game Logic Module (RL Edition)
Online learning loop: observe -> policy decides -> act -> wait reward -> update.
The robot starts random (like an untrained cat) and improves via real owner feedback.
"""
import time
import logging
import threading
import random
import numpy as np
from typing import Optional, Callable, Dict
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class GameState(Enum):
    IDLE = auto()
    OBSERVING = auto()           # legacy: kept for back-compat
    OBSERVING_LEFT = auto()      # peek left, recognize what's there
    OBSERVING_RIGHT = auto()     # peek right, recognize what's there
    THINKING = auto()            # legacy: kept for back-compat
    TENTATIVE = auto()           # legacy: fake-out hesitation
    DECIDING = auto()            # comparing identifications vs target
    HIT_LEFT = auto()
    HIT_CENTER = auto()
    HIT_RIGHT = auto()
    NOT_FOUND = auto()           # target not seen on either side, abort round
    WAITING_FEEDBACK = auto()    # waiting owner correct/wrong feedback
    UPDATING = auto()            # stats update (no real backprop in inference mode)
    CORRECT = auto()
    WRONG = auto()
    PAUSED = auto()              # between rounds


@dataclass
class GameConfig:
    correct_reward: float = 1.0
    wrong_reward: float = -0.1
    auto_advance: bool = False      # if True, auto start next round after delay
    advance_delay: float = 3.0
    max_rounds: int = 0             # 0 = unlimited
    enable_fakeout: bool = True     # dramatic hesitation even when policy is confident


class ZhenbangGame:
    """
    Online RL game loop. The policy network decides left/right based on raw pixels.
    Owner provides sparse reward (+1 / -0.1). Network updates immediately.
    """

    def __init__(
        self,
        robot_actor,
        vision,
        voice,
        trainer,              # REINFORCETrainer instance (kept for stats/reward update)
        detector=None,        # ObjectDetector — open-vocab detection (Phase B)
        workspace=None,       # kinematics.WorkspaceMap — Phase B continuous IK
        vla_runner=None,      # vla_runner.VLARunner — Phase H learned policy
        cap_agent=None,       # cap_agent.PickAgent — Phase I Tool-Use mode
        cap_codegen=None,     # cap_codegen.CodeAgent — Phase I-A real CaP (LLM writes Python)
        config: Optional[GameConfig] = None,
        state_callback: Optional[Callable] = None,
    ):
        self.robot = robot_actor
        self.vision = vision
        self.voice = voice
        self.trainer = trainer
        self.detector = detector
        self.workspace = workspace
        self.vla_runner = vla_runner
        self.cap_agent = cap_agent
        self.cap_codegen = cap_codegen
        self.cfg = config or GameConfig()
        self.state_callback = state_callback

        # Backend selection — UI can override at runtime.
        # "auto" picks best available: codegen > cap > vla > perception
        # "codegen" → cap_codegen (real CaP, LLM writes Python)
        # "cap" → cap_agent (Tool Use)
        # "vla" → vla_runner (Phase H)
        # "perception" → Phase B detect+rules
        self.force_backend: str = "auto"

        # VLA episode timing
        self.vla_max_seconds: float = 12.0      # max episode duration
        self.vla_control_hz: float = 20.0       # action send rate (matches training)

        # VLA telemetry (read by UI for live visualization)
        from collections import deque
        self.vla_status: str = "idle"           # idle / planning / executing / done
        self.vla_step_count: int = 0
        self.vla_total_steps: int = 0
        self.vla_instruction: str = ""
        self.vla_last_action: Optional[Dict[str, float]] = None
        self.vla_action_history = deque(maxlen=60)   # (step, action_dict) tuples
        self.vla_episode_start_t: Optional[float] = None

        # CaP (Phase I) telemetry — read by UI
        self.cap_status: str = "idle"           # idle / planning / executing / reflecting / done
        self.cap_task: str = ""
        self.cap_plan: list = []                # list of "primitive(args)" strings
        self.cap_current_step: str = ""
        self.cap_current_step_idx: int = 0
        self.cap_step_log = deque(maxlen=40)    # (timestamp, step_name, info) tuples
        self.cap_episode_start_t: Optional[float] = None
        self.cap_last_outcome: str = ""
        self.cap_last_code: str = ""            # codegen mode: LLM-generated Python

        self.state = GameState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._feedback_event = threading.Event()
        self._feedback_result: Optional[bool] = None  # True=correct, False=wrong
        self._start_round_event = threading.Event()
        self._running = False
        self._paused = False

        # Round tracking
        self.round_count = 0
        self.last_choice_str: Optional[str] = None
        self.last_probs: Optional[object] = None  # numpy array
        self.last_reward: Optional[float] = None
        self.last_advantage: Optional[float] = None
        self.current_ground_truth: str = "left"
        self.current_target_idx: int = 0
        self.last_target_idx: int = 0
        # Phase B+: free-text target overrides target_idx if non-empty.
        # Used by the "type any object" UI input.
        self.current_target_text: Optional[str] = None
        self.last_target_text: Optional[str] = None
        # Per-round perception state.
        # last_detections: full list of {name_idx, name, bbox, score} from the round's
        # one observation pass. The thinking panel renders straight from this.
        self.last_detections: Optional[list] = None
        self.last_target_detection: Optional[Dict] = None  # winning detection for target
        self.last_decision_text: Optional[str] = None
        # Legacy fields, kept for back-compat with anything older still reading them
        self.last_left_id: Optional[Dict] = None
        self.last_right_id: Optional[Dict] = None

    def set_target(self, target_idx: int):
        """Preset target by index (0=萝卜, 1=纸巾, 2=可乐). Clears free-text target."""
        self.current_target_idx = int(target_idx)
        self.current_target_text = None
        logger.info("Round target set (preset idx=%d)", target_idx)

    def set_target_text(self, text: str):
        """Free-text target — robot will look for any object the user types."""
        text = (text or "").strip()
        if not text:
            self.current_target_text = None
            return
        self.current_target_text = text
        logger.info("Round target set (free text): %r", text)

    def _notify_state(self):
        if self.state_callback:
            try:
                self.state_callback(self.state, {
                    "round": self.round_count,
                    "eps": self.trainer.current_eps if self.trainer else 0,
                    "acc": self.trainer.current_accuracy if self.trainer else 0,
                    "baseline": self.trainer.baseline if self.trainer else 0,
                    "reward": self.last_reward,
                    "choice": self.last_choice_str,
                })
            except Exception as e:
                logger.error("State callback error: %s", e)

    def _transition(self, new_state: GameState):
        logger.info("State: %s -> %s", self.state.name, new_state.name)
        self.state = new_state
        self._notify_state()

    def set_ground_truth(self, side: str):
        """Owner tells us where the carrot currently is (so we know if robot was correct)."""
        self.current_ground_truth = side
        logger.info("Ground truth set: carrot is on %s", side)

    def start_game(self):
        if self._running:
            return "已经在运行中"
        self._stop_event.clear()
        self._feedback_event.clear()
        self._start_round_event.set()
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._game_loop, daemon=True)
        self._thread.start()
        logger.info("Game loop started.")
        return "游戏已启动！机器人开始从奖励中学习。"

    def stop_game(self):
        self._stop_event.set()
        self._feedback_event.set()
        self._start_round_event.set()
        self.robot.stop()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._transition(GameState.IDLE)
        logger.info("Game loop stopped.")
        return "已停止"

    def pause(self):
        self._paused = True
        self._start_round_event.clear()
        logger.info("Paused. Owner can swap items or inspect model.")
        return "已暂停（可交换物品位置）"

    def resume_next_round(self):
        """Owner signals they are ready for next round (items placed)."""
        self._paused = False
        self._start_round_event.set()
        logger.info("Resuming next round.")
        return "继续下一轮"

    def give_feedback(self, is_correct: bool):
        """Owner presses correct/wrong button."""
        self._feedback_result = is_correct
        self._feedback_event.set()

    def _game_loop(self):
        while not self._stop_event.is_set():
            if self._paused:
                self._transition(GameState.PAUSED)
                time.sleep(0.2)
                continue

            # Wait for owner to signal ready (or auto-advance)
            if not self.cfg.auto_advance:
                self._start_round_event.wait(timeout=1.0)
                if not self._start_round_event.is_set() and not self._paused:
                    continue
                self._start_round_event.clear()

            if self.cfg.max_rounds > 0 and self.round_count >= self.cfg.max_rounds:
                logger.info("Reached max rounds.")
                break

            try:
                self._run_one_round()
            except Exception as e:
                logger.exception("Round error: %s", e)
                time.sleep(1.0)

        self._running = False
        self._transition(GameState.IDLE)

    def _run_one_round(self):
        """
        Round dispatch — pick the execution backend.

        force_backend overrides auto-selection. Order of preference (auto):
          codegen > cap > vla > perception
        """
        backend = self.force_backend or "auto"

        if backend == "codegen" and self.cap_codegen is not None:
            return self._run_one_round_codegen()
        if backend == "cap" and self.cap_agent is not None:
            return self._run_one_round_cap()
        if backend == "vla" and self.vla_runner is not None:
            return self._run_one_round_vla()
        if backend == "perception":
            return self._run_one_round_perception()

        # auto — prefer real CaP codegen, but ONLY if it has an LLM provider
        # (codegen has no hardcoded fallback, so without LLM it's a dead branch).
        # Order: codegen+LLM > cap (works with hardcoded plans too) > vla > perception
        if self.cap_codegen is not None and getattr(self.cap_codegen, "llm", None):
            return self._run_one_round_codegen()
        if self.cap_agent is not None:
            return self._run_one_round_cap()
        if self.vla_runner is not None:
            return self._run_one_round_vla()
        return self._run_one_round_perception()

    # ------------------------------------------------------------------
    def _run_one_round_vla(self):
        """
        Phase H: SmolVLA closed-loop control.
        Camera frame + state + instruction → VLA → joint commands @ 20Hz.
        """
        robot = self.robot
        vision = self.vision
        voice = self.voice
        runner = self.vla_runner

        target_idx = self.current_target_idx
        target_text = self.current_target_text
        self.last_target_idx = target_idx
        self.last_target_text = target_text
        self.last_detections = None
        self.last_target_detection = None
        self.last_choice_str = None

        # Resolve instruction
        from vla_runner import map_target_to_instruction
        if target_text:
            instruction = map_target_to_instruction(target_text)
            target_label = target_text
        else:
            preset_zh = ("萝卜", "纸巾", "可乐")[target_idx] if 0 <= target_idx < 3 else "?"
            instruction = map_target_to_instruction(preset_zh)
            target_label = preset_zh

        self.last_decision_text = f"VLA 指令: \"{instruction}\""
        logger.info("VLA round | target=%s instruction='%s'", target_label, instruction)

        # Init VLA telemetry for UI
        self.vla_instruction = instruction
        self.vla_step_count = 0
        self.vla_total_steps = int(self.vla_max_seconds * self.vla_control_hz)
        self.vla_last_action = None
        self.vla_action_history.clear()
        self.vla_episode_start_t = time.time()
        self.vla_status = "preparing"

        # ---- PREPARE: home pose ----
        self._transition(GameState.OBSERVING)
        robot.reset_stop()
        home_pose = robot.poses.get("home")
        if home_pose:
            robot.go_to_pose(home_pose, duration=1.0, steps=20)
        time.sleep(0.4)

        # ---- VLA inference loop ----
        self._transition(GameState.HIT_CENTER)   # generic "executing" state
        runner.reset()
        # (sound effects disabled — they were distracting; UI shows status instead)
        self.vla_status = "executing"

        dt = 1.0 / self.vla_control_hz
        max_steps = self.vla_total_steps
        t_start = time.time()
        step_count = 0

        try:
            while step_count < max_steps and not self._stop_event.is_set():
                t_step = time.time()
                frame = vision.read_frame() if vision else None
                if frame is None:
                    time.sleep(dt)
                    continue
                robot._sync_joints_from_robot()
                state_dict = dict(robot.current_joints)

                action = runner.get_action(frame, state_dict, instruction)

                # Telemetry update
                step_count += 1
                self.vla_step_count = step_count
                self.vla_last_action = dict(action)
                self.vla_action_history.append((step_count, dict(action)))

                # Send to robot
                robot.go_to_pose(action, duration=dt * 0.9, steps=2)

                # Pace
                elapsed = time.time() - t_step
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except Exception as e:
            logger.exception("VLA inference loop error: %s", e)

        self.vla_status = "done"
        wall = time.time() - t_start
        logger.info("VLA episode done: %d steps in %.1fs (target %.1fs)",
                    step_count, wall, self.vla_max_seconds)

        # ---- WAITING_FEEDBACK ----
        self._transition(GameState.WAITING_FEEDBACK)
        feedback_received = self._feedback_event.wait(timeout=60.0)
        is_correct = self._feedback_result if feedback_received else False
        self._feedback_event.clear()
        self._feedback_result = None

        reward = self.cfg.correct_reward if is_correct else self.cfg.wrong_reward
        self.last_reward = reward
        self.last_choice_str = "vla"

        # Stats only
        self._transition(GameState.UPDATING)
        if self.trainer is not None:
            advantage = reward - self.trainer.baseline
            self.last_advantage = advantage
            # Action idx is meaningless here; pass 0
            self.trainer.update(frame, target_idx, 0, 0.0, reward)

        # ---- REACT (no sound, no celebration motion — go straight home) ----
        if is_correct:
            self._transition(GameState.CORRECT)
        else:
            self._transition(GameState.WRONG)

        # Return to home for next round
        if home_pose:
            robot.go_to_pose(home_pose, duration=1.2, steps=24)

        self.round_count += 1
        time.sleep(0.5)

        if self.cfg.auto_advance:
            time.sleep(self.cfg.advance_delay)
        else:
            self._paused = True
            self._start_round_event.clear()

    # ------------------------------------------------------------------
    def _run_one_round_cap(self):
        """
        Phase I: Code-as-Policies + Agent layer.
        LLM (or hardcoded) plan → closed-loop primitives (visual servo, grasp,
        verify, retry) → reflection on failure.

        Reuses Phase B detector + workspace; no learned policy needed.
        """
        robot = self.robot
        agent = self.cap_agent

        target_idx = self.current_target_idx
        target_text = self.current_target_text
        self.last_target_idx = target_idx
        self.last_target_text = target_text
        self.last_detections = None
        self.last_target_detection = None
        self.last_choice_str = None

        # Resolve task description
        if target_text:
            task = target_text
            target_label = target_text
        else:
            preset_zh = ("萝卜", "纸巾", "可乐")[target_idx] if 0 <= target_idx < 3 else "萝卜"
            task = preset_zh
            target_label = preset_zh

        self.last_decision_text = f"CaP 任务: \"{task}\""
        logger.info("CaP round | target=%s task=%r", target_label, task)

        # Init CaP telemetry
        self.cap_task = task
        self.cap_plan = []
        self.cap_current_step = ""
        self.cap_current_step_idx = 0
        self.cap_step_log.clear()
        self.cap_episode_start_t = time.time()
        self.cap_status = "preparing"
        self.cap_last_outcome = ""

        # Hook agent's state callback into our telemetry
        def _agent_cb(step: str, info: str):
            self.cap_status = step
            self.cap_current_step = info if info else step
            self.cap_step_log.append((time.time(), step, info))
            if step == "plan" and info:
                # plan info is "step1 → step2 → ..."
                self.cap_plan = info.split(" → ")
            elif step == "exec":
                # info format: "step k/N: prim_name"
                self.cap_current_step_idx += 1
        agent.state_callback = _agent_cb

        # ---- PREPARE: home pose, clear stop flag ----
        self._transition(GameState.OBSERVING)
        robot.reset_stop()
        agent.reset()
        home_pose = robot.poses.get("home")
        if home_pose:
            robot.go_to_pose(home_pose, duration=1.0, steps=20)
        time.sleep(0.3)

        # ---- EXECUTE ----
        self._transition(GameState.HIT_CENTER)   # generic "executing" state
        self.cap_status = "executing"

        # Run agent in this thread (blocking) — but check _stop_event between phases
        # The agent itself supports cancellation via .cancel()
        try:
            # Start a watchdog to propagate _stop_event into agent.cancel()
            stop_watcher_running = threading.Event()
            stop_watcher_running.set()

            def _watch_stop():
                while stop_watcher_running.is_set():
                    if self._stop_event.is_set():
                        agent.cancel()
                        break
                    time.sleep(0.1)

            watcher = threading.Thread(target=_watch_stop, daemon=True)
            watcher.start()

            success, info = agent.run(task)
            stop_watcher_running.clear()

        except Exception as e:
            logger.exception("CaP agent.run error: %s", e)
            success, info = False, f"exception: {e}"

        wall = time.time() - self.cap_episode_start_t
        self.cap_status = "done"
        self.cap_last_outcome = "success" if success else "failed"
        logger.info("CaP episode done: success=%s info=%s elapsed=%.1fs",
                    success, info, wall)

        # ---- WAITING_FEEDBACK ----
        # Even though agent self-reports success, owner has the final say
        # (e.g. agent thinks grip succeeded by angle but visually it's wrong)
        self._transition(GameState.WAITING_FEEDBACK)
        feedback_received = self._feedback_event.wait(timeout=60.0)
        is_correct = self._feedback_result if feedback_received else success
        self._feedback_event.clear()
        self._feedback_result = None

        reward = self.cfg.correct_reward if is_correct else self.cfg.wrong_reward
        self.last_reward = reward
        self.last_choice_str = "cap"

        # Stats
        self._transition(GameState.UPDATING)
        if self.trainer is not None:
            advantage = reward - self.trainer.baseline
            self.last_advantage = advantage
            # No frame snapshot here; pass None
            try:
                self.trainer.update(None, target_idx, 0, 0.0, reward)
            except Exception:
                pass

        if is_correct:
            self._transition(GameState.CORRECT)
        else:
            self._transition(GameState.WRONG)

        # Return to home for next round
        if home_pose:
            robot.go_to_pose(home_pose, duration=1.2, steps=24)

        self.round_count += 1
        time.sleep(0.4)

        if self.cfg.auto_advance:
            time.sleep(self.cfg.advance_delay)
        else:
            self._paused = True
            self._start_round_event.clear()

    # ------------------------------------------------------------------
    def _run_one_round_codegen(self):
        """
        Phase I-A: Real Code-as-Policies. LLM generates executable Python,
        sandbox-execute. Closest to Liang et al. 2022.
        """
        robot = self.robot
        agent = self.cap_codegen

        target_idx = self.current_target_idx
        target_text = self.current_target_text
        self.last_target_idx = target_idx
        self.last_target_text = target_text
        self.last_detections = None
        self.last_target_detection = None
        self.last_choice_str = None

        # Resolve task
        if target_text:
            task = target_text
            target_label = target_text
        else:
            preset_zh = ("萝卜", "纸巾", "可乐")[target_idx] if 0 <= target_idx < 3 else "萝卜"
            task = preset_zh
            target_label = preset_zh

        self.last_decision_text = f"CodeGen 任务: \"{task}\""
        logger.info("CodeGen round | target=%s task=%r", target_label, task)

        # Init telemetry
        self.cap_task = task
        self.cap_plan = []
        self.cap_current_step = ""
        self.cap_current_step_idx = 0
        self.cap_step_log.clear()
        self.cap_episode_start_t = time.time()
        self.cap_status = "preparing"
        self.cap_last_outcome = ""
        self.cap_last_code = ""

        def _agent_cb(step: str, info: str):
            self.cap_status = step
            self.cap_current_step = info or step
            self.cap_step_log.append((time.time(), step, info))
        agent.state_callback = _agent_cb

        # Prepare
        self._transition(GameState.OBSERVING)
        robot.reset_stop()
        agent.reset()
        home_pose = robot.poses.get("home")
        if home_pose:
            robot.go_to_pose(home_pose, duration=1.0, steps=20)
        time.sleep(0.3)

        # Execute
        self._transition(GameState.HIT_CENTER)
        self.cap_status = "executing"

        try:
            stop_watcher_running = threading.Event()
            stop_watcher_running.set()

            def _watch_stop():
                while stop_watcher_running.is_set():
                    if self._stop_event.is_set():
                        agent.cancel()
                        break
                    time.sleep(0.1)

            watcher = threading.Thread(target=_watch_stop, daemon=True)
            watcher.start()

            success, info = agent.run(task)
            stop_watcher_running.clear()

            # Stash generated code for UI
            self.cap_last_code = agent.last_code

        except Exception as e:
            logger.exception("CodeGen run error: %s", e)
            success, info = False, f"exception: {e}"

        wall = time.time() - self.cap_episode_start_t
        self.cap_status = "done"
        self.cap_last_outcome = "success" if success else "failed"
        logger.info("CodeGen episode done: success=%s info=%s elapsed=%.1fs",
                    success, info, wall)

        # Feedback / stats — same as cap_agent round
        self._transition(GameState.WAITING_FEEDBACK)
        feedback_received = self._feedback_event.wait(timeout=60.0)
        is_correct = self._feedback_result if feedback_received else success
        self._feedback_event.clear()
        self._feedback_result = None

        reward = self.cfg.correct_reward if is_correct else self.cfg.wrong_reward
        self.last_reward = reward
        self.last_choice_str = "codegen"

        self._transition(GameState.UPDATING)
        if self.trainer is not None:
            advantage = reward - self.trainer.baseline
            self.last_advantage = advantage
            try:
                self.trainer.update(None, target_idx, 0, 0.0, reward)
            except Exception:
                pass

        if is_correct:
            self._transition(GameState.CORRECT)
        else:
            self._transition(GameState.WRONG)

        if home_pose:
            robot.go_to_pose(home_pose, duration=1.2, steps=24)

        self.round_count += 1
        time.sleep(0.4)

        if self.cfg.auto_advance:
            time.sleep(self.cfg.advance_delay)
        else:
            self._paused = True
            self._start_round_event.clear()

    # ------------------------------------------------------------------
    def _run_one_round_perception(self):
        """
        Phase A/B: VLM detection + rule-based / RBF execution.
        Used when no VLA runner is loaded.
        """
        robot = self.robot
        vision = self.vision
        voice = self.voice
        trainer = self.trainer
        detector = self.detector

        target_idx = self.current_target_idx
        target_text = self.current_target_text   # free-text override (None if preset)
        self.last_target_idx = target_idx
        self.last_target_text = target_text
        self.last_detections = None
        self.last_target_detection = None
        self.last_decision_text = None
        self.last_choice_str = None
        self.last_left_id = None
        self.last_right_id = None

        if target_text:
            target_label = target_text
        else:
            target_label = ("萝卜", "纸巾", "可乐")[target_idx] if 0 <= target_idx < 3 else "?"

        # ---- PREPARE: center pose ----
        self._transition(GameState.OBSERVING)
        robot.reset_stop()
        center_pose = robot.poses.get("center_look") or robot.poses.get("home")
        if center_pose:
            robot.go_to_pose(center_pose, duration=0.8, steps=16)
        time.sleep(0.4)

        # ---- OBSERVE: one full-frame detection pass ----
        if voice:
            voice.play_emotion("observe")
        frame = vision.read_frame() if vision else None
        detections = []
        target_det = None

        if frame is not None and detector is not None:
            try:
                if target_text:
                    # Open-vocab single-target search
                    target_det = detector.detect_target(frame, target_text)
                    if target_det is not None:
                        detections = [target_det]
                        logger.info("Detected '%s' @ %s score=%.2f",
                                    target_text, target_det["bbox"], target_det["score"])
                    else:
                        logger.info("Free-text target '%s' not found in frame", target_text)
                else:
                    # Preset 3-class detection
                    detections = detector.detect(frame, top_k_per_class=2)
                    logger.info("Detections: %d found", len(detections))
                    for d in detections:
                        logger.info("  - %s @ %s  score=%.2f",
                                    d["name"], d["bbox"], d["score"])
            except Exception as e:
                logger.exception("Detection failed: %s", e)
        self.last_detections = detections

        time.sleep(0.6)

        # ---- DECIDE: pick best detection matching the target ----
        self._transition(GameState.DECIDING)
        if voice:
            voice.play_emotion("think")

        if target_text is None:
            target_det = detector.best_for_target(detections, target_idx) if detector else None
        # else: target_det already set above
        self.last_target_detection = target_det

        action_idx = -1
        side = None
        world_xy = None    # Phase B: (x, y) in [0,1]² of calibrated workspace
        if target_det is None:
            self.last_decision_text = f"画面里没看到 {target_label}，跳过本轮"
        else:
            x1, y1, x2, y2 = target_det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            frame_w = frame.shape[1] if frame is not None else 640

            if self.workspace is not None:
                # Phase B: pixel → world (x,y) via homography → continuous IK
                try:
                    wx, wy = self.workspace.pixel_to_world(cx, cy)
                    world_xy = (wx, wy)
                    side = "world"
                    action_idx = 0   # not used in Phase B; kept for stats
                    self.last_decision_text = (
                        f"检测到 {target_det['name']} ({target_det['score']*100:.0f}%) "
                        f"像素 ({cx:.0f},{cy:.0f}) → 桌面 ({wx:.2f},{wy:.2f})"
                    )
                except Exception as e:
                    logger.exception("Phase B mapping failed: %s", e)

            if world_xy is None:
                # Phase A fallback: 3-zone discrete
                if cx < frame_w / 3:
                    side, action_idx = "left",   0
                elif cx > frame_w * 2 / 3:
                    side, action_idx = "right",  2
                else:
                    side, action_idx = "center", 1
                self.last_decision_text = (
                    f"检测到 {target_det['name']} ({target_det['score']*100:.0f}%) "
                    f"在 x={cx:.0f}/{frame_w} → 拍{side}"
                )

        logger.info("DECISION: target=%s | %s", target_label, self.last_decision_text)

        self.last_choice_str = side
        # 3-action probability vector [left, center, right]
        score = target_det["score"] if target_det else 0.0
        self.last_probs = np.array([
            score if side == "left"   else 0.0,
            score if side == "center" else 0.0,
            score if side == "right"  else 0.0,
        ])
        log_prob_val = float(np.log(0.5))
        time.sleep(0.4)

        # ---- NOT FOUND: abort round ----
        if action_idx == -1:
            self._transition(GameState.NOT_FOUND)
            logger.info("Round %d | NOT_FOUND: target=%s not on table", self.round_count + 1, target_label)
            if voice:
                voice.play_emotion("confused")
            robot.hesitate(cycles=2)
            time.sleep(0.6)
            self.round_count += 1
            if self.cfg.auto_advance:
                time.sleep(self.cfg.advance_delay)
            else:
                self._paused = True
                self._start_round_event.clear()
            return

        # ---- HIT ----
        if world_xy is not None:
            # Phase B: pick a "side" for state display based on x position only
            wx = world_xy[0]
            if wx < 0.33:    hit_state = GameState.HIT_LEFT
            elif wx > 0.67:  hit_state = GameState.HIT_RIGHT
            else:            hit_state = GameState.HIT_CENTER
        else:
            hit_state = {
                "left":   GameState.HIT_LEFT,
                "center": GameState.HIT_CENTER,
                "right":  GameState.HIT_RIGHT,
            }[side]
        self._transition(hit_state)
        logger.info("Round %d | %s", self.round_count + 1, self.last_decision_text)
        if voice:
            voice.play_emotion("act")
        if world_xy is not None:
            robot.tap_at_world(world_xy[0], world_xy[1], workspace=self.workspace)
        else:
            robot.quick_tap(side=side)
        time.sleep(0.3)

        # ---- WAITING_FEEDBACK ----
        self._transition(GameState.WAITING_FEEDBACK)
        feedback_received = self._feedback_event.wait(timeout=60.0)
        is_correct = self._feedback_result if feedback_received else (side == self.current_ground_truth)
        self._feedback_event.clear()
        self._feedback_result = None

        reward = self.cfg.correct_reward if is_correct else self.cfg.wrong_reward
        self.last_reward = reward

        # ---- UPDATING (stats only in inference mode) ----
        self._transition(GameState.UPDATING)
        last_frame = frame
        if last_frame is not None and trainer is not None:
            advantage = reward - trainer.baseline
            self.last_advantage = advantage
            trainer.update(last_frame, target_idx, action_idx, log_prob_val, reward)

        # ---- REACT ----
        if is_correct:
            self._transition(GameState.CORRECT)
            if voice:
                voice.play_emotion("excited")
            robot.celebrate()
        else:
            self._transition(GameState.WRONG)
            if voice:
                voice.play_emotion("sad", block=True)
            robot.sad_reaction()

        self.round_count += 1
        time.sleep(0.5)

        if self.cfg.auto_advance:
            time.sleep(self.cfg.advance_delay)
        else:
            self._paused = True
            self._start_round_event.clear()

    @property
    def state_name(self) -> str:
        return self.state.name if self.state else "UNKNOWN"

    def get_status_dict(self) -> Dict:
        return {
            "round": self.round_count,
            "state": self.state_name,
            "last_choice": self.last_choice_str or "-",
            "last_reward": self.last_reward,
            "last_advantage": self.last_advantage,
            "accuracy": self.trainer.current_accuracy if self.trainer else 0,
            "baseline": self.trainer.baseline if self.trainer else 0,
            "eps": self.trainer.current_eps if self.trainer else 0,
            "total_reward": self.trainer.total_reward if self.trainer else 0,
        }
