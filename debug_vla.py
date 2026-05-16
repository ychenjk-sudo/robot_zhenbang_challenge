"""
诊断 VLA 推理为什么不 work：
  1. action 输出的数值范围是不是 unnormalized 的真实关节角
  2. camera2/3 应该传零还是复制 camera1
"""
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import cv2
import numpy as np
import torch
from vla_runner import VLARunner, JOINT_NAMES

print("=" * 60)
print("加载 VLA …")
runner = VLARunner()
runner.reset()
print()

# 用录的真实图作为输入
frame = cv2.imread("/tmp/test_real.jpg")
if frame is None:
    # fallback: 第一帧训练数据
    frame = cv2.imread("demos/episode_0000/frames/frame_00000.jpg")
print(f"测试图: shape={frame.shape}")

# 用录第一帧时的 follower 关节角 (从 states.npy 拿)
states = np.load("demos/episode_0000/states.npy")
state_dict = dict(zip(JOINT_NAMES, states[0].tolist()))
print(f"输入 state: {state_dict}")
print()

print("=" * 60)
print("Test A: camera2/3 = 复制 camera1 (现状)")
print("=" * 60)
# 临时跑当前实现
action_a = runner.get_action(frame, state_dict, "pick up the carrot")
print(f"输出 action: {action_a}")
print()

# 看几个连续 step (chunk 缓存有效)
print("=" * 60)
print("连续 5 步 action (看变化幅度) …")
print("=" * 60)
for i in range(5):
    a = runner.get_action(frame, state_dict, "pick up the carrot")
    diffs = {k: round(a[k] - state_dict[k], 2) for k in JOINT_NAMES}
    print(f"step {i+1}: action={ {k: round(v,2) for k,v in a.items()} }")
    print(f"          delta from state: {diffs}")
print()

print("=" * 60)
print("Test B: 改 camera2/3 = 零 张量")
print("=" * 60)
# Hack: 修改 _build_observation 让 camera2/3 = 0
import torch as T
runner.reset()
orig_build = runner._build_observation
def build_with_zero_cams(frame_bgr, joint_state, instruction):
    obs = orig_build(frame_bgr, joint_state, instruction)
    obs["observation.images.camera2"] = T.zeros_like(obs["observation.images.camera1"])
    obs["observation.images.camera3"] = T.zeros_like(obs["observation.images.camera1"])
    return obs
runner._build_observation = build_with_zero_cams

action_b = runner.get_action(frame, state_dict, "pick up the carrot")
print(f"输出 action: {action_b}")
print()
print("=" * 60)
print("对比 (是否相同):")
for j in JOINT_NAMES:
    same = abs(action_a[j] - action_b[j]) < 0.001
    print(f"  {j}: A={action_a[j]:+.2f}  B={action_b[j]:+.2f}  {'同' if same else '不同'}")
print()

# 训练时 episode 0 的 action[0] 是多少, 对比看模型输出有多远
true_action = np.load("demos/episode_0000/actions.npy")[0]
print("=" * 60)
print("训练数据里这一帧的真实 action (ground truth):")
print(f"  {dict(zip(JOINT_NAMES, [round(v,2) for v in true_action]))}")
print()
print("VLA 应该跟上面的值接近。差距越大 → 模型越不对。")
