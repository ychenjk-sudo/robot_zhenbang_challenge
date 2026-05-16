"""
诊断 VLA 模型在训练数据上能否复现 ground-truth 轨迹.
用 ep_0034 (新数据) 的几个时刻喂模型, 看预测的 action 跟真实 action 差多少.

也对比 camera2/3 = zeros vs replicate camera1, 看哪个更准.
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.WARNING)

import numpy as np
import cv2
import torch

from vla_runner import VLARunner, JOINT_NAMES

# Test on ep_0034 at multiple time points
EP = "demos/episode_0034"
TEST_FRAMES = [0, 60, 100, 120, 140]   # 0s, 3s, 5s, 6s, 7s

print("加载模型...")
runner = VLARunner()
print()

states = np.load(f"{EP}/states.npy")
actions = np.load(f"{EP}/actions.npy")

def predict_with_camera_mode(frame_idx, mode):
    """mode: 'zeros' or 'replicate'"""
    runner.reset()
    img_bgr = cv2.imread(f"{EP}/frames/frame_{frame_idx:05d}.jpg")
    state = dict(zip(JOINT_NAMES, states[frame_idx].tolist()))

    # Hack: temporarily monkey-patch _build_observation to use the chosen mode
    orig_build = runner._build_observation
    def custom_build(frame_bgr, joint_state, instruction):
        obs = orig_build(frame_bgr, joint_state, instruction)
        if mode == "replicate":
            obs["observation.images.camera2"] = obs["observation.images.camera1"].clone()
            obs["observation.images.camera3"] = obs["observation.images.camera1"].clone()
        # else: zeros (default)
        return obs
    runner._build_observation = custom_build

    pred = runner.get_action(img_bgr, state, "pick up the carrot and place it on the left")

    runner._build_observation = orig_build
    return pred


print("=" * 90)
print("测试: 在训练数据 ep_0034 上复现 ground-truth")
print("=" * 90)
print()

for fi in TEST_FRAMES:
    t = fi / 20.0
    gt = dict(zip(JOINT_NAMES, actions[fi].tolist()))
    state = dict(zip(JOINT_NAMES, states[fi].tolist()))

    print(f"━━━ 时刻 t={t:.1f}s (frame {fi}) ━━━")
    print(f"  当前 state:        pan={state['shoulder_pan']:+6.1f} lift={state['shoulder_lift']:+6.1f} grip={state['gripper']:+5.1f}")
    print(f"  ★ Ground-truth:    pan={gt['shoulder_pan']:+6.1f} lift={gt['shoulder_lift']:+6.1f} grip={gt['gripper']:+5.1f}")

    for mode in ["zeros", "replicate"]:
        pred = predict_with_camera_mode(fi, mode)
        # diff from gt
        diff = np.array([abs(pred[j] - gt[j]) for j in JOINT_NAMES])
        flag = "✓" if diff.mean() < 5 else ("⚠" if diff.mean() < 15 else "✗")
        print(f"  {flag} cam2/3={mode:<10}: pan={pred['shoulder_pan']:+6.1f} lift={pred['shoulder_lift']:+6.1f} grip={pred['gripper']:+5.1f}  | mean diff = {diff.mean():.1f}°")
    print()
