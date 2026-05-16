# 模型迭代历史

> 记录 2026-05-04 至 2026-05-14 的架构演进、训练数据、失败根因。

---

## Phase 0 — CLIP + REINFORCE 在线学习

**时间**: 2026-05-04 早期  
**对应挑战**: 萝卜真棒（已归档）

| 维度 | 方案 |
|------|------|
| 感知 | CLIP 零样本 + 19 张照片微调 |
| 决策 | CNN 策略网络 (3×84×84 → 2 logits) |
| 执行 | 手写姿态脚本，三段离散位置（左/中/右） |
| 训练 | REINFORCE 策略梯度，ε-greedy 探索 |
| 动作空间 | 2 个（拍左 / 拍右） |

**结果** ❌ CLIP 零样本不足以区分同类物体。用户换成真的萝卜、卷纸、可乐后，CLIP 先验知识完全不对。REINFORCE 收敛慢，离散动作空间太粗糙。

---

## Phase A — 三段离散规则

**时间**: 2026-05-04 中午  
**对应挑战**: 萝卜真棒（已归档）

| 维度 | 方案 |
|------|------|
| 感知 | 无（手选物品） |
| 决策 | 规则：当前物品 → 选择对应的三段位置之一 |
| 执行 | 三段离散 IK 位置 |

**结果** ⏭ 快速过度的调试阶段，无实际使用价值。

---

## Phase B — 连续工作空间 + VLM 检测

**时间**: 2026-05-04 ~ 2026-05-05  
**对应挑战**: Pick and Place（架构基础）

| 维度 | 方案 |
|------|------|
| 感知 | GLM-4V-Flash → OWL-ViT → **Qwen2.5-VL-7B**（最终选定） |
| 空间定位 | 9 点 RBF 标定 → 连续 IK 工作空间 |
| 决策 | 规则：VLM 返回 bbox → 投影世界坐标 → IK 解算 |
| 执行 | `go_to_pose` 插值下降 + hand-coded 夹爪时序 |
| 动作空间 | 连续 x,y 平面位置 |

**关键突破**:
- 感知链路首次跑通，Qwen2.5-VL 能准确区分萝卜/可乐/纸巾
- RBF 连续 IK 替代三段离散，动作空间从 2 个扩展到连续平面

**局限**:
- 无 learned action model——执行靠手写规则脚本
- 夹爪时序全靠 `time.sleep` 参数猜

---

## Phase H — SmolVLA 端到端（已废弃）

**时间**: 2026-05-05 ~ 2026-05-06  
**对应挑战**: Pick and Place（实验性，已替代）

| 维度 | 方案 |
|------|------|
| 模型 | **SmolVLA-450M**（端到端 VLA） |
| 底座 | SmolVLM2-500M-Video-Instruct |
| VLM 层数 | 27 → 16（为 M2 16GB 裁剪） |
| 训练数据 | 46 条 demo（pick_carrot ×17, push_cola ×14, pull_tissue ×15） |
| 训练量 | 6000 step，batch_size=4，8GB+ checkpoint |
| Loss | 0.59 → **0.176** |
| 推理 | MPS，action chunk=50，20Hz |
| 训练时间 | ~8 小时 |

**结果** ❌ 实机失败。

**失败根因分析**:
- 46 条 demo 中大部分是同一固定位置
- 动作关节与视频帧模态未对齐（用户原话）
- 实机测试出现：**高频抖动**、**不下落**（停在物体上方 1-2cm）、**空夹**（夹爪未碰到物体就闭合）
- 450M 参数对 M2 16GB 来说过于沉重

**退役原因**:
- Diffusion Policy（~50M）参数量少 10 倍
- DP 训练时间仅 30 分钟（SmolVLA 需要 8 小时）
- DP 与上层 CaP 解耦，失败不会拖垮整个系统

---

## Phase I — Code as Policy（当前核心）

**时间**: 2026-05-07 ~ 至今  
**对应挑战**: Pick and Place

| 维度 | 方案 |
|------|------|
| 决策层 | **LLM（Qwen API）写 Python 代码** |
| 原语层 | 手写 primitive（approach / grasp_with_retry / transport_to / release / go_home） |
| 硬件层 | LeRobot SO-101 + Feetech |

### I-A: CodeGen（LLM 写代码）
LLM 现场生成 Python 代码运行在沙盒中：
```python
def task():
    pos = api.locate("萝卜")
    api.hover_above(pos[0], pos[1])
    api.grasp_with_retry("萝卜")
    api.transport_to(LEFT)
    api.open_gripper()
    api.go_home()
    return True
```
- 解释性语言 → `for` 循环、`try/except`、条件判断
- 沙盒安全：AST 白名单 + 100 iter 上限 + 30s 超时

### I: Tool Use（LLM 选 primitive）
LLM 从预设 primitives 列表中选择组合，不走沙盒：
- 更安全，但不能现场发明新动作
- 用于低风险场景

**成果**: 连续 2 轮 100% 命中率（offset=-0.09）

---

## Phase D — Diffusion Policy（当前动作执行）

**时间**: 2026-05-14 ~ 至今  
**对应挑战**: Pick and Place

| 维度 | DP v2（当前使用） |
|------|------------------|
| 模型 | Diffusion Policy (DDPM, epsilon) |
| 参数量 | ~50M |
| 骨干网络 | ResNet18 + UNet [256,512,1024] |
| 训练数据 | **5 条** 遥操 pick-only demo |
| 位置覆盖 | 右侧、左侧、中间靠你、中间靠远、右斜 30° |
| 总帧数 | 594 帧（~30 秒） |
| 训练步数 | 2000 step |
| Loss | ~0.05 |
| 训练时间 | ~30 分钟 |
| 推理速度 | <20ms/步（MPS） |

### 架构集成

```
VLM → (cx, cy) → CaP → (图像 + 关节角 + 指令) → DP → [关节序列] → 机器人
```

- CaP 负责：VLM 检测 + LLM 规划 + transport/release/home
- DP 负责：下降 → 抓取 → 抬升 的精细动作

### DP v1 vs v2

| 版本 | 训练数据 | 结果 |
|------|---------|------|
| v1 | 35 条同位置 pick-place | ❌ 只学会固定轨迹，无法泛化 |
| v2 | 5 条多样化位置 pick-only | ✅ 多样性覆盖更好，可验证泛化 |

---

## 架构决策树

```
想要什么任务？
├── "识别萝卜位置并拍击" → Phase 0 RL CNN（已归档）
├── "用手写规则拿物品"  → Phase I (CaP Tool Use)
├── "用 LLM 灵活编排"  → Phase I-A (CodeGen)
└── "用学习轨迹执行"    → Phase D (DP)
                           └── 先用 CaP 定位到 hover
                           　　 └── DP 执行下降 + 抓取 + 抬升
                           　　　   └── CaP 执行 transport + release + home
```
