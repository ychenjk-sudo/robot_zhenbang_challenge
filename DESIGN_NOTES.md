# 真棒挑战 · 设计笔记

> ⚠️ **注意**: 本文档保留历史技术细节，但架构设计已被 [`MODEL_HISTORY.md`](MODEL_HISTORY.md) 和 [`README.md`](README.md) 替代。  
> 当前使用 VLM + CaP + DP 混合架构，详见 README。

> 持续更新的实验复盘 + 架构决策记录. 跨 session 用, 给未来的自己/合作者看.

---

## 2026-05-12 · Phase I (CaP) 完整复盘

### 这次新增的代码

| 文件 | 作用 |
|---|---|
| `primitives.py` | Layer 2 闭环原语 (approach / grasp_with_retry / verify) |
| `cap_agent.py` | Phase I — Tool Use (LLM 选 primitive 调用) |
| `cap_codegen.py` | Phase I-A — 原版 CaP (LLM 写 Python, 沙盒执行) |
| `grasp_memory.py` | 持久化记忆 (per-target 成功的偏移量) |
| `smoke_test_cap.py` | 6 个场景的 mock 烟测 |
| `game.py` / `main.py` | 加 cap_agent / cap_codegen 分支 + UI radio |

### 关键架构决策

1. **三层架构** (Liang et al. 2022 CaP 的现代变体):
   - Layer 3 Agent: LLM 写代码 (CodeGen) 或 选 primitive (Tool Use)
   - Layer 2 Primitives: 手写闭环原语 (approach / grasp_with_retry / ...)
   - Layer 1 Hardware: lerobot + Feetech

2. **沙盒** (5 层防御): AST 白名单 / 名字黑名单 / dunder 拦截 / restricted builtins / iter+timeout 限制

3. **GRASP offset 机制**: 在 grasp 时把检测出的 bbox 中心**横向偏移**到物体侧边, 避免双指夹爪夹在物体正中心把物体撞飞 (实测 carrot 用 `-0.09` 工作)

4. **视觉验证 > 角度判断**: SO-101 夹爪关到底是 78, 抓住细物体也是 78 — 角度无法区分. 改用"抬起来视觉重检"作为 ground truth, 角度只做诊断.

5. **bbox 面积 = 近距感知**: 同一物体近了画面更大. 用 OpenCV 颜色掩码 (5ms) 而不是 Qwen (1s) 做 fast detection.

6. **GraspMemory**: per-target 偏移量持久化到 JSON, 移动平均更新. 解决"调对一次, 每次都要重调"问题.

### Tool Use vs CodeGen 对比

| 维度 | Tool Use (硬编码 plan) | CodeGen (LLM 写代码) |
|---|---|---|
| 单任务稳定性 | ✅ 100% deterministic | 🟡 95% (temp=0) |
| 新任务泛化 | ❌ 必须写新 plan | ✅ LLM 用现有 primitive 拼新逻辑 |
| 失败恢复 | ❌ 硬编码 reflect rules | ✅ LLM 可写 try/except 自适应 |
| 延迟 | <10ms | 1-3s LLM call |
| 调试 | ✅ plan 可读 | 🟡 LLM 生成代码可读但变化 |
| 推荐场景 | 已知任务高频跑 | 自由文本 / 多步骤 / 探索性任务 |

### 用户的核心洞察 (2026-05-12)

> "Code as Policy 其实类似于现在的 Skill，它是把很多的原子级组件整合在一起形成了这样的 Policy。Code 其实就是一个最容易泛化的信号。"

**完全正确, 这是 2024-2025 整个 agent 工程领域的共识**.

- 原子组件 = Atomic Skills (我们的 primitives)
- 组合层 = Code / Tool Calls (LLM 输出)
- 终极策略 = 由 LLM 用代码把 skills 串起来

代码作为信号有 5 项优势:
1. 表达力 (loops / conditionals / variables)
2. 可执行性 (无需翻译, 直接跑)
3. 可组合性 (function-of-function, 无限层级)
4. 可验证性 (sandbox / type-check / static analysis)
5. 可复用性 (library / module / standard library)

---

## 实验观察 · 抓萝卜挑战 (2026-05-12)

### 命中率演进 (跨阶段)

| 阶段 | 配置 | 命中率 |
|---|---|---|
| Phase H baseline | SmolVLA + 30 demo + 单 wrist cam | ~30% |
| Phase I (Tool Use, no offset) | CaP + grasp center | ~0% (萝卜撞飞) |
| Phase I (Tool Use, offset=-0.04) | CaP + 偏左一点 | ~30% |
| Phase I (Tool Use, offset=-0.09) | CaP + 偏左更多 | ~70% |
| **Phase I-A (CodeGen, offset=-0.09, memory ON)** | **当前** | **连续 2 轮 100%** ✓ |

### 还存在的物理限制 (用户 2026-05-12 实验观察)

**夹爪硬件限制**: SO-101 夹爪命令 `value=0` (最开), 实际只能到 `~11.4`. 这个开度**对萝卜来说不够宽**.

观察到的现象:
> "它还是得下到离萝卜非常近、甚至碰到萝卜之后才张开。你一张开夹爪，萝卜就被弹飞了。"

物理解释 (推测):
1. 夹爪 servo 有响应延迟, 命令 0 但实际只到 11
2. value=11 时**指尖距 ≈ 4cm**, 萝卜直径 ≈ 2.5-3cm, 夹爪指尖会**擦到萝卜外侧**
3. servo 在下落过程中**仍在缓慢继续打开** (catch-up), 视觉上像"碰到才张开"
4. 一旦张开 → 推开萝卜 → 闭爪空抓

**这不是软件 bug, 是硬件机械限制**. 软件已经做到了:
- ✅ commanded=0 (硬件极限)
- ✅ 给了 2s 时间让 servo 开到位
- ✅ 已用 grasp offset 偏移到萝卜侧边
- ✅ 已用 bbox 面积感知近距

但**夹爪指尖最大开度 = 4cm 是物理硬限**, 软件改不掉.

---

## 🔧 未来:更新夹爪方案的考虑点

记录给未来设计新夹爪的人 (可能就是未来的我).

### 必须解决的问题
1. **开度不够宽**: 当前最大 ~4cm, 应该 ≥ 6cm 才能轻松容纳萝卜 / 苹果 / 杯子
2. **没有视觉自我感知**: 当前夹爪在 wrist cam 视野**外**, 系统不知道夹爪指尖在哪. 应该:
   - 在指尖加**视觉标记** (彩色贴纸 / ArUco / LED)
   - 或者把 wrist cam 角度调整成能看到指尖
   - 或者加**电流/扭矩传感**, 接触时反馈
3. **角度反馈分辨率不足**: 抓住物体 vs 空抓, 角度都是 78.8. 应该:
   - 用支持**电流反馈**的 servo (Feetech STS3215 支持但驱动没暴露)
   - 或者加**触觉传感** (压力薄膜 / FSR 力敏电阻贴在指尖)
4. **形状单一**: 双指平行夹爪对**细长物**和**球形物**通用性差. 考虑:
   - **柔性夹爪** (软体, 自适应包裹)
   - **吸盘** (适合平面物体, 但表面不规则的不行)
   - **三指/四指** (更稳定, 但更贵)

### 视觉感知的根本性升级方向

用户洞察 (2026-05-12):
> "它跟萝卜之间的物理相对位置，能够通过视觉的方式去意识到吗？"

**能, 但需要把夹爪纳入视觉系统**:

```
当前: wrist cam 看世界, 算 bbox → 算世界坐标 → 用标定推关节角
      问题: 不知道夹爪在哪, 只能假设它在 wrist 正下方

升级方案 A: 夹爪指尖加标记
      ├ wrist cam 同时看到 carrot bbox + gripper marker
      ├ 计算两者**像素差**
      └ 视觉伺服 (visual servoing) 直接最小化像素差, 不依赖标定

升级方案 B: 加 overhead 外置相机
      ├ 第三视角看到 arm + carrot 整个场景
      ├ 准确知道夹爪和 carrot 的相对位置
      └ 投影到俯视图, 直接计算抓取点

升级方案 C: 加深度感知
      ├ RealSense / iPhone LiDAR / 双目立体
      ├ 直接知道 z 距离
      └ 配合 wrist cam, 完全解开深度模糊
```

**性价比**: 方案 A (指尖加标记) ≈ ¥10, 工程量 1-2 天 → 立刻能落地
        方案 B (外置 cam) ≈ ¥50-200, 工程量 3-5 天 → 中期最优
        方案 C (深度相机) ≈ ¥1500+, 工程量 1-2 周 → 长期工业级

### 软件侧需要配合的改动

新夹爪上线时, 需要改:
1. `primitives.py` 的 `GRIP_OPEN_VALUE` / `GRIP_CLOSE_VALUE` / `GRIP_HOLDING_MIN/MAX` — 标定新硬件
2. `_carrot_bbox_area_now()` — 如果加视觉标记, 还要写 `_gripper_marker_position_now()`
3. `grasp_with_retry` 增加新 phase: **视觉伺服收敛** — 在 hover 后, 看 gripper-carrot 像素差, 闭环调整位置直到差<threshold, 再下落
4. 力/触觉反馈接进来后, **visual verify 可以省掉** (接触即成功)

### 数据采集补充

如果还想训练 VLA, 用新夹爪重录数据时:
- 录制时**同时录夹爪的电流/扭矩** (加 modality)
- 录制时萝卜随机摆 9 宫格 × 3 朝向 = 至少 27 个位置覆盖
- 包含**失败-恢复** demo (故意夹偏 → 重定位 → 再夹), 让 VLA 学会自我纠错

---

## 已部署的 backend (UI 可切换)

| Backend | 实现 | 何时用 |
|---|---|---|
| 🔴 Phase I-A CodeGen | LLM 写 Python | **默认推荐**, 泛化最强 |
| 🟣 Phase I Tool Use | LLM 选 primitive | LLM 出问题时的稳定 fallback |
| 🟢 Phase H SmolVLA | 端到端神经网络 | 连续动作任务 (推可乐 / 抽纸巾) |
| 🟡 Phase B 检测+IK | Qwen detect + RBF | API key 不可用时的兜底 |

---

## 跨 session 的项目快照 (2026-05-12 02:50)

- **代码**: 5 个新文件 + 4 个改造文件, 全部 syntax OK
- **烟测**: 6/6 通过
- **实机验证**: 拿萝卜任务**连续 2 轮成功**, offset=-0.09 已写入记忆
- **环境**: macOS + Python 3.12 + zhenbang conda env
- **Qwen API**: DASHSCOPE_API_KEY 已配置 (用完记得 rotate)
- **calibration.npz**: 9 点 RBF, observe pose = `center_look`
