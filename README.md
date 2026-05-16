# 🤖 萝卜纸巾 · 真棒挑战 · 在线强化学习版

> **核心理念**: 机器人像小猫一样从零开始学。它没有先验知识，不知道什么是萝卜什么是纸巾——
> 它只学到 **"看到某种画面 → 拍左边/右边 → 获得主人的奖励"**。
>
> 猫不是因为认识萝卜而答对，而是因为答对能得到冻干。  
> 机器人不是因为认识物品而答对，而是因为答对能听到"真棒"。

---

## 🧠 为什么这是真正的"学会"，而不是脚本

你之前的直觉是对的：很多机器人"训练"其实是预编程——开发者把"萝卜=橙色"先写进代码里，机器人只是执行既定规则。

但真正的猫（和这个机器人）的学习过程是：

| 阶段 | 猫 | 这个机器人 |
|------|----|-----------|
| **初识世界** | 不知道萝卜/纸巾是什么 | 神经网络权重完全随机 |
| **做动作** | 随机拍一个 | ε-greedy策略：50%随机，50%按网络输出 |
| **获得反馈** | 对了=冻干，错了=没有 | 对了=**奖励+1**，错了=**惩罚-0.1** |
| **更新信念** | 大脑强化"那个视觉→拍那边"的神经连接 | **REINFORCE策略梯度**：当场反向传播更新CNN权重 |
| **逐渐变准** | 20轮后几乎不猜错 | 准确率从50%上升到90%+ |

**关键点**: 机器人从来不"识别"物品。它学到的是一个 **视觉→动作→奖励** 的映射函数。你把萝卜和纸巾换成香蕉和袜子，它一样能重新学会——只要奖励信号还在。

---

## 🛠️ 硬件要求

| 设备 | 型号示例 | 说明 |
|------|---------|------|
| 机械臂 | **SO-ARM100 / SO-101 / Koch** | LeRobot 兼容的 6DOF+夹爪 |
| 摄像头 | USB 摄像头 (640×480即可) | 原始像素输入神经网络 |
| 电脑 | Ubuntu / macOS / Windows | 能跑 PyTorch 的笔记本即可 |
| 桌子 | 稳固桌面 | 臂展范围内 |

> 💡 不需要GPU！小型CNN在CPU上推理+训练每轮 < 0.2秒。

---

## 📦 安装

### 1. 环境

```bash
conda create -n zhenbang python=3.10
conda activate zhenbang
```

### 2. 安装 LeRobot (控制机械臂)

```bash
git clone https://github.com/huggingface/lerobot
cd lerobot
pip install -e ".[feetech]"   # SO-ARM100/SO-101
# pip install -e ".[dynamixel]"  # Koch
```

### 3. 安装本项目

```bash
cd ~/your_workspace
unzip robot_zhenbang_challenge.zip
cd robot_zhenbang_challenge
pip install -r requirements.txt   # torch, opencv, gradio, matplotlib, pygame
```

### 4. 配置姿态 (非常重要!)

编辑 `config.yaml` 中的 `game.poses`。用 LeRobot 遥操作找到适合你桌面的角度：

```bash
# 在 lerobot 目录下运行
conda activate zhenbang
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

记录关节角度填入 `config.yaml`。

---

## 🚀 运行

```bash
conda activate zhenbang
cd ~/your_workspace/robot_zhenbang_challenge
python main.py
```

浏览器打开 `http://localhost:7860`

---

## 🎮 交互流程（每一轮都是一次训练样本）

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 摆放物品          萝卜放左边？还是随机换位置？主人决定！       │
│      ↓                                                          │
│  2. 点击"开始下一轮"                                             │
│      ↓                                                          │
│  3. 🎥 机器人拍照 → CNN输出概率 [p_left, p_right]                │
│      ↓                                                          │
│  4. 🤖 犹豫/假动作（戏剧效果，时间长短和网络不确定性成正比）      │
│      ↓                                                          │
│  5. 👋 拍击（左或右）                                            │
│      ↓                                                          │
│  6. ⏸️ 等待你反馈                                                │
│      ↓                                                          │
│  7. 你点 ✅（奖励+1）或 ❌（惩罚-0.1）                            │
│      ↓                                                          │
│  8. 🧠 当场反向传播！网络权重更新！                              │
│      ↓                                                          │
│  9. 🎉 庆祝 / 😿 沮丧 动作                                       │
│      ↓                                                          │
│  10. 回到步骤1，你可以交换物品位置防止机器人"位置作弊"            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 学习曲线解读

界面右下角的图表实时显示：

- **绿色/红色散点**: 每一轮的奖励（+1绿色 / -0.1红色）
- **蓝色曲线**: 最近20轮的滑动窗口准确率（从50%开始，目标90%+）
- **橙色填充**: 探索率 ε 的衰减（从0.5降到0.05，越学越自信）

**预期学习进度**（在固定位置训练）：
- **0~10轮**: 随机瞎猜，准确率 ~50%
- **15~30轮**: 开始捕捉到视觉模式，准确率上升到 ~70%
- **40~60轮**: 基本稳定，准确率 ~85-95%

**进阶**: 一旦固定位置学会了，点 **"交换物品位置"**，准确率会掉回60%左右——因为机器人必须真正学会"看内容"而不是"记位置"。再训练20轮，它会重新稳定。

---

## 🧪 可做的实验

### 实验1: 位置作弊测试
- 训练20轮（萝卜始终左边）→ 准确率应该上升
- 突然点击 **"交换物品位置"** → 准确率暴跌
- 继续训练20轮 → 准确率重新上升
- **结论**: 机器人最终学会的是"视觉内容"而非"位置"

### 实验2: 泛化到新物品
- 把萝卜纸巾换成 **苹果 + 遥控器**
- 加载之前保存的模型（对萝卜纸巾训练的）
- 观察准确率：如果跌到50%，说明它真的只学到了特定视觉模式
- 从头训练20轮 → 重新学会
- **结论**: 不存在"通用物体识别"，只有"特定视觉-动作-奖励映射"

### 实验3: 冻结底层，只训顶层
- 修改 `learner.py` 只更新最后一层FC
- 观察是否学得更快/更慢
- **结论**: 探索不同网络结构的学习效率

### 实验4: 改变奖励大小
- 修改 `config.yaml` 里的 `correct_reward` / `wrong_reward`
- 比如 +5 / -1，观察学习速度变化

---

## 🗂️ 项目结构

```
robot_zhenbang_challenge/
├── config.yaml              # 机械臂参数 + 学习超参数 + 姿态
├── main.py                  # Gradio UI + 全局协调
├── learner.py               # ⭐ 核心: CNN策略网络 + REINFORCE训练器
├── game.py                  # ⭐ 核心: RL交互循环 (观察→决策→奖励→更新)
├── robot_actor.py           # 机械臂动作编排 (犹豫、拍击、庆祝、沮丧)
├── vision.py                # 摄像头 + 帧预处理 (原始像素→网络输入)
├── voice.py                 # 语音反馈 (真棒/不对/思考)
├── teleop_guide.py          # 姿态标定指南
├── requirements.txt
├── README.md
├── checkpoints/             # 自动保存的模型
│   └── zhenbang_policy.pt
└── sounds/
    ├── carrot_zhenbang.mp3
    ├── tissue_zhenbang.mp3
    ├── wrong.mp3
    ├── hello_think.mp3
    └── tap.mp3
```

---

## 🔬 算法细节

### 网络架构

```
Input: (3, 84, 84) RGB desk image [0,1]
  ↓ Conv2d(3→16, 5×5, stride=2) + ReLU   → (16, 42, 42)
  ↓ Conv2d(16→32, 3×3, stride=2) + ReLU  → (32, 21, 21)
  ↓ Conv2d(32→64, 3×3, stride=2) + ReLU  → (64, 11, 11)
  ↓ Flatten → 7744-d
  ↓ Linear(7744 → 128) + ReLU
  ↓ Linear(128 → 2) logits
  ↓ Softmax → [p_left, p_right]
```

~1M 参数，CPU 推理 < 10ms。

### REINFORCE with Baseline

每一轮交互后执行单次梯度更新：

```
log_prob = log( π(action | image) )
baseline = EMA(reward_history)   # 指数移动平均
advantage = reward - baseline

loss = -log_prob * advantage - entropy_coef * H(π)
optimizer.step()
```

### ε-Greedy 探索

```
初始 ε = 0.5  (50%概率忽略网络，完全随机探索)
每轮衰减 → 最终 ε = 0.05
防止过早收敛到局部最优
```

---

## 💾 模型保存/加载

界面有按钮：
- **保存模型**: 存到 `checkpoints/zhenbang_policy.pt`，包含网络权重 + 优化器状态 + 基线值 + 已训练轮数
- **加载模型**: 恢复之前训好的策略
- **重置学习**: 丢弃所有权重，回到随机猜测状态（从零开始）

---

## ⚠️ 安全提示

- 先用调试面板逐个测试姿态
- 紧急停止按钮随时可用
- 确保摄像头视野能同时看到两个物品
- 初始阶段机器人会频繁猜错（正常的！），别让它的拍击动作打到脆弱物品

---

## 🏷️ License

MIT License

**Made with 🤖 + 🐱 + 🍖 + ∇**  
Enjoy watching your robot learn like a cat!
