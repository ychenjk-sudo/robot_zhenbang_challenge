# 🥕 萝卜·纸巾·可乐 — 真棒挑战

VLM + Code as Policy + Diffusion Policy 混合架构的实物机械臂控制。

## 架构

```
VLM (Qwen2.5-VL) → 物体检测 (像素坐标)
    ↓ (wx, wy)
CaP (Code as Policy) → LLM 写代码编排任务
    ↓ (图像 + 关节角 + 指令)
DP (Diffusion Policy) → 50 步动作轨迹
    ↓ [6 关节目标位置]
LeRobot SO-101 → Feetech 伺服器执行
```

## 仓库内容

```
robot_zhenbang_challenge/
├── main.py           # Gradio UI + 入口
├── game.py           # 游戏循环 + 状态机
├── primitives.py     # 手写原语 (approach/grasp/transport/…)
├── dp_runner.py      # Diffusion Policy 推理封装
├── cap_codegen.py    # Code as Policy LLM 代码生成
├── cap_agent.py      # Tool-Use 模式原语调用
├── robot_actor.py    # 机械臂控制 (go_to_pose / send_action)
├── detector.py       # VLM 检测器 (Qwen API)
├── kinematics.py     # 运动学 / RBF 工作空间映射
├── vision.py         # 摄像头读取
├── voice.py          # 语音输入 (FunASR)
├── voice_input.py    # 语音输入处理
├── grasp_memory.py   # 抓取偏移量持久化记忆
├── stats_tracker.py  # 命中率统计
├── calibrate.py      # 9点RBF标定
├── convert_to_lerobot.py  # Demo → LeRobot 数据集
├── record_demos.py   # 遥操录制 demo
├── sounds/           # 音效
├── demos/            # 5 条多样化 Pick-only demo
│   ├── episode_0036  # 萝卜右侧
│   ├── episode_0037  # 萝卜左侧
│   ├── episode_0038  # 萝卜中间靠近
│   ├── episode_0040  # 萝卜中间靠远
│   └── episode_0041  # 萝卜右斜 30°
├── config.yaml       # 配置
└── DESIGN_NOTES.md   # 架构设计文档
```

## 模型权重

仓库源码不含模型权重（~290MB）。从 Release 下载：

```bash
# 下载 DP 模型
gh release download v1.0 --repo ychenjk-sudo/robot_zhenbang_challenge
# 或手动: https://github.com/ychenjk-sudo/robot_zhenbang_challenge/releases/tag/v1.0
```

解压到 `checkpoints/diffusion_zhenbang_v2/checkpoints/002000/pretrained_model/`

## 在新电脑上部署

### 必须手动操作

| # | 步骤 | 说明 |
|---|------|------|
| 1 | `pip install -e "git+https://github.com/huggingface/lerobot#egg=lerobot[feetech]"` | 安装 LeRobot + Feetech 驱动 |
| 2 | 下载 Release `model.safetensors` 放到 `checkpoints/.../` | DP 模型权重 |
| 3 | 配置 `DASHSCOPE_API_KEY` | VLM 和 LLM 需要 Qwen API |
| 4 | 运行 `python calibrate.py` 做 9 点 RBF 标定 | 每台桌子都需要 |
| 5 | 摄像头连接 | USB 摄像头或机械臂自带 cam |
| 6 | 配置 `config.yaml` 的 `robot.port` 和 `camera.index` | 根据硬件调整 |

### 快速启动（标定 + API 就绪后）

```bash
conda activate zhenbang
cd robot_zhenbang_challenge
python main.py
```

浏览器打开 `http://localhost:7860`。

### 重新训练 DP（可选）

```bash
python convert_to_lerobot.py --input demos --repo-id local/zhenbang_pickplace_dp \
    --root datasets
lerobot-train --policy.type diffusion --dataset.repo_id local/zhenbang_pickplace_dp \
    --steps 2000
```

## 硬件要求

- SO-ARM100 / SO-101 机械臂（LeRobot 兼容）
- USB 摄像头
- macOS / Linux（未在 Windows 测试）
- 无需 GPU（MPS / CPU 均可运行推理）

## 依赖

- Python 3.10+
- LeRobot (HuggingFace)
- PyTorch
- Gradio (UI)
- FunASR (语音)
- DASHSCOPE API Key

## 已删除的旧文件

清理仓库时移除了以下不再需要的历史文件：

- `debug_*.py` — 实验性调试脚本
- `vla_runner.py` — 废弃的 SmolVLA 推理（被 DP 替代）
- `smoke_test*.py` — 早期烟测
- `learner.py` — 废弃的在线 RL 训练器（REINFORCE CNN）
- `migrate_to_new_user.sh` — 历史迁移脚本（一次性）
- `teleop_guide.py` — 早期引导脚本
- `generate_sounds.py` — 音效生成（一次性）
- `test_gripper_marker.py`, `test_perception.py` — 实验性测试
- `monitor_training.py`, `train.sh` — 实验脚本
- `requirements.txt` — 用 `pyproject.toml` 依赖管理替代
- `demos_old/`, `demos_old_state_eq_action/` — 旧版 demo 数据（36 条旧轨迹 + 实验数据）
- `demos/episode_0000` ~ `0035` — 旧版 36 条同位置 pick-place 轨迹
- `demos/episode_0039` — 录制不合格的单条
- `offline_data/` — 离线测试图片
- `marker_test_*.png` — 指尖标记测试图
- `calibration.npz` — 旧标定文件（每环境重标）

## License

MIT
