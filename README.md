# 🥕 萝卜·纸巾·可乐 — 机器人抓取挑战

本项目包含两个不同阶段的机械臂控制挑战，共享同一套硬件（SO-ARM100 + USB 摄像头）。

---

## 🧩 Challenge 1: 萝卜真棒（已归档）

> 原版挑战。2026-05-04 ~ 2026-05-05。

模拟小猫学习模式的小型 RL 实验：机械臂不知道什么是萝卜/纸巾，通过试错 + 奖励信号学会"看到橙色 → 拍左边"。

| 维度 | 方案 |
|------|------|
| 感知 | CLIP 零样本 + 19 张照片微调 |
| 决策 | CNN 策略网络 (3×84×84 → 2 logits) |
| 执行 | 三段离散位置（左/中/右），手写姿态脚本 |
| 训练 | REINFORCE 在线学习，ε-greedy 探索，每轮权重更新 |
| 模型 | `learner.py` 中的小型 CNN（~1M 参数） |
| 数据 | 无预训练数据，在线上通过训练生成 |

**相关脚本**（已删除，见 git history）：

| 文件 | 作用 |
|------|------|
| `learner.py` | CNN 策略网络 + REINFORCE 训练器 |
| `teleop_guide.py` | 姿态标定引导 |

**限制**：零样本 CLIP 精度不足，REINFORCE 收敛慢，三段离散动作空间太粗糙，无法泛化到不同物体位置。

---

## 🚀 Challenge 2: Pick and Place（当前架构）

> 2026-05-05 至今。泛化为通用抓取框架。

**VLM + Code as Policy + Diffusion Policy 混合架构**：感知用 VLM，规划用 LLM 写代码，动作用 DP。

### 架构总览

```
VLM (Qwen2.5-VL)          → 检测物体 (像素坐标 (cx, cy))
    ↓
CaP (Code as Policy)      → LLM 写 Python 代码编排任务
    ↓ (图像 + 关节角 + 指令)
DP (Diffusion Policy)     → 50 步动作轨迹 (6 维关节序列)
    ↓ [关节目标位置]
LeRobot SO-101 (Feetech)  → 伺服器执行
```

### 模块分工

| 层级 | 模块 | 作用 |
|------|------|------|
| **感知** | `detector.py` + Qwen API | VLM 检测物体，返回像素坐标 |
| **定位** | `kinematics.py` + `calibrate.py` | 9 点 RBF 标定，像素→世界坐标映射 |
| **规划** | `cap_codegen.py` + Qwen API | LLM 写 Python 代码（locate → grasp → transport → release） |
| **规划** | `cap_agent.py` | Tool Use 模式：LLM 选 primitive 组合 |
| **执行** | `primitives.py` | 手写原语（approach / grasp_with_retry / transport_to / release / go_home） |
| **动作** | `dp_runner.py` + Diffusion Policy | 学习精细抓取轨迹（替代手写 approach + grasp） |
| **控制** | `robot_actor.py` | 机械臂底层控制 (go_to_pose / send_action) |
| **记忆** | `grasp_memory.py` | 抓取偏移量持久化 |
| **统计** | `stats_tracker.py` | 命中率统计 |
| **UI** | `main.py` + Gradio | 可视化操作界面 |
| **数据** | `record_demos.py` | 遥操录制 demo |
| **数据** | `convert_to_lerobot.py` | Demo → LeRobot 数据集 |

### 模型

| 模型 | 文件 | 说明 |
|------|------|------|
| **DP v2（当前使用）** | Release v1.0 | 5 条多样化 pick demo 训练。CaP 送到 hover 后，DP 执行下降→抓取→抬升 |
| **SmolVLA（已废弃）** | Release v1.0-smolvla | 46 条 demo 训练的端到端 VLA。因数据不足 + 参数量级不匹配 M2 16GB 退役 |

### 训练迭代

详细记录见 [`MODEL_HISTORY.md`](MODEL_HISTORY.md)。

---

## 🛠️ 硬件要求

| 设备 | 型号 |
|------|------|
| 机械臂 | SO-ARM100 / SO-101 / Koch（LeRobot 兼容） |
| 摄像头 | USB 摄像头（腕部或顶部视角） |
| 电脑 | macOS / Linux，16GB+ 内存 |
| 桌子 | 稳固桌面 |

> 💡 无需 GPU！DP 推理在 MPS / CPU 上均可运行。

---

## 📦 在新电脑上部署

### 1. 安装 LeRobot

```bash
git clone https://github.com/huggingface/lerobot
cd lerobot
pip install -e ".[feetech]"
```

### 2. 下载本仓库 + 模型

```bash
git clone https://github.com/ychenjk-sudo/robot_zhenbang_challenge
cd robot_zhenbang_challenge

# 下载 DP 模型
gh release download v1.0
# 创建 checkpoints 目录并解压
mkdir -p checkpoints/diffusion_zhenbang_v2/checkpoints/002000/pretrained_model
mv model.safetensors config.json policy_preprocessor.json policy_postprocessor.json \
   checkpoints/diffusion_zhenbang_v2/checkpoints/002000/pretrained_model/
```

### 3. 配置 API Key

```bash
export DASHSCOPE_API_KEY="sk-your-key"
```

### 4. 标定

```bash
python calibrate.py
```

### 5. 运行

```bash
python main.py
```

浏览器打开 `http://localhost:7860`。

---

## ⚠️ 新手注意事项

### 🖥️ 摄像头/权限

- macOS 首次运行需要 **系统设置 → 隐私与安全性 → 摄像头 → 终端授权**
- 不要用 `conda run` 启动，用 `conda activate` + 直接 `python main.py`
- `config.yaml` 中的 `camera.index`：0=外接 USB, 1=内置 iSight

### 🔌 机械臂

- 两个 USB 同时连接：leader（手操器）和 follower（执行臂）
- 端口名在 `/dev/cu.usbmodem*`，`config.yaml` 正确配置 `port`
- 首次使用前必须做 **9 点 RBF 标定** (`python calibrate.py`)

### ☁️ API

- VLM + CaP 均依赖 **DASHSCOPE API**（阿里通义千问），需充值
- API 欠费时 CaP 会退化为手写 plan（无 LLM 能力），DP 本地推理不受影响
- 免费额度用完后训练/检测会报 `401` / `Arrearage`

### 🗃️ 数据

- 5 条 demo 在 `demos/` 中，用于复现 DP 训练
- 完整 LeRobot 数据集需要运行 `python convert_to_lerobot.py --input demos --repo-id local/zhenbang_pickplace_dp_v3 --root datasets`
- 录制新模式：`python record_demos.py`

### 🧪 训练

- DP 训练命令（2000 step, ~30 分钟）:
  ```bash
  lerobot-train --policy.type diffusion --dataset.repo_id local/zhenbang_pickplace_dp_v3 --steps 2000
  ```
- 运行 `main.py` 时，Phase D 状态应显示 `🔵 Phase D: Diffusion Policy 已加载`
- 如果 Phase D 未激活，检查：
  1. 模型文件是否在正确路径
  2. `--dataset.root` 是否指向正确的数据集目录

### 🔒 安全

- 运行前先手动测试每个姿态
- 紧急情况按 `Ctrl+C` 停掉终端
- 机械臂动作范围有限，确保桌面无阻挡
- 初始测试时把萝卜放离夹爪远端，避免碰撞

---

## 📖 项目文件索引

| 文件 | 属于哪个 Challenge | 说明 |
|------|-------------------|------|
| `main.py` | Pick and Place (当前) | Gradio UI + 入口 |
| `game.py` | Pick and Place | 游戏循环 + 状态机 |
| `primitives.py` | Pick and Place | 手写原语 |
| `dp_runner.py` | Pick and Place | DP 推理封装 |
| `cap_codegen.py` | Pick and Place | CaP 代码生成 |
| `cap_agent.py` | Pick and Place | CaP Tool Use |
| `robot_actor.py` | 共用 | 机械臂控制 |
| `detector.py` | Pick and Place | VLM 检测器 |
| `kinematics.py` | Pick and Place | RBF 运动学 |
| `calibrate.py` | Pick and Place | 9 点标定 |
| `vision.py` | 共用 | 摄像头 |
| `convert_to_lerobot.py` | Pick and Place | 数据转换 |
| `record_demos.py` | Pick and Place | 遥操录制 |
| `grasp_memory.py` | Pick and Place | 抓取记忆 |
| `stats_tracker.py` | 共用 | 统计 |
| `voice.py` / `voice_input.py` | 共用 | 语音输入 |

---

## License

MIT
