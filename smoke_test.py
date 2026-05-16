"""
SmolVLA smoke test — 快速验证：
  1. 你录的 demo 数据完整性 / 数值合理性
  2. SmolVLA 能不能加载
  3. 用你的一帧数据跑一次前向推理 → 输出 action 维度对不对

通过后再继续录完 50 条；不通过我们换 ACT。

用法:
    python smoke_test.py                  # 完整流程
    python smoke_test.py --data-only      # 只检查数据 (不加载 SmolVLA)
    python smoke_test.py --plot           # 额外画一条 episode 的轨迹图
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Phase 1: data integrity
# ---------------------------------------------------------------------------

def check_data(demos_dir: Path) -> dict:
    print("=" * 64)
    print("PHASE 1: 数据完整性检查")
    print("=" * 64)

    eps = sorted(demos_dir.glob("episode_*"))
    print(f"  Episodes 总数: {len(eps)}")
    if not eps:
        raise RuntimeError(f"{demos_dir} 没有 episode_* 子目录")

    summary = {
        "n_episodes": len(eps),
        "total_frames": 0,
        "by_task": {},
        "frame_counts": [],
        "state_stats": None,
        "action_stats": None,
        "issues": [],
    }
    all_states = []
    all_actions = []

    for ep_dir in eps:
        meta_path = ep_dir / "meta.json"
        states_path = ep_dir / "states.npy"
        actions_path = ep_dir / "actions.npy"
        frames_dir = ep_dir / "frames"

        # Required files
        for p in (meta_path, states_path, actions_path):
            if not p.exists():
                summary["issues"].append(f"{ep_dir.name}: 缺 {p.name}")

        try:
            meta = json.loads(meta_path.read_text())
            states = np.load(states_path)
            actions = np.load(actions_path)
        except Exception as e:
            summary["issues"].append(f"{ep_dir.name}: 加载失败 {e}")
            continue

        n_meta = meta.get("n_frames", 0)
        n_states = states.shape[0]
        n_actions = actions.shape[0]
        n_frames = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0

        if not (n_meta == n_states == n_actions == n_frames):
            summary["issues"].append(
                f"{ep_dir.name}: 长度不一致 meta={n_meta} states={n_states} "
                f"actions={n_actions} frames={n_frames}")

        if states.shape[1] != 6:
            summary["issues"].append(f"{ep_dir.name}: states 列数 {states.shape[1]} ≠ 6")
        if actions.shape[1] != 6:
            summary["issues"].append(f"{ep_dir.name}: actions 列数 {actions.shape[1]} ≠ 6")
        if np.isnan(states).any() or np.isinf(states).any():
            summary["issues"].append(f"{ep_dir.name}: states 含 NaN/Inf")
        if np.isnan(actions).any() or np.isinf(actions).any():
            summary["issues"].append(f"{ep_dir.name}: actions 含 NaN/Inf")
        if (states == 0).all():
            summary["issues"].append(f"{ep_dir.name}: states 全 0（采集没生效？）")

        summary["total_frames"] += n_frames
        summary["frame_counts"].append(n_frames)
        task = meta.get("task_name", "?")
        summary["by_task"][task] = summary["by_task"].get(task, 0) + 1
        all_states.append(states)
        all_actions.append(actions)

    if all_states:
        S = np.vstack(all_states)
        A = np.vstack(all_actions)
        summary["state_stats"] = {
            "min": S.min(axis=0).tolist(),
            "max": S.max(axis=0).tolist(),
            "mean": S.mean(axis=0).tolist(),
            "std":  S.std(axis=0).tolist(),
        }
        summary["action_stats"] = {
            "min": A.min(axis=0).tolist(),
            "max": A.max(axis=0).tolist(),
            "mean": A.mean(axis=0).tolist(),
            "std":  A.std(axis=0).tolist(),
        }

    # ---- print summary ----
    print(f"  总帧数: {summary['total_frames']}")
    fc = summary["frame_counts"]
    if fc:
        print(f"  每条帧数 (s={20}fps 假设): min={min(fc)}({min(fc)/20:.1f}s) "
              f"max={max(fc)}({max(fc)/20:.1f}s) "
              f"mean={int(np.mean(fc))}({np.mean(fc)/20:.1f}s)")
    print(f"  任务分布: {summary['by_task']}")

    if summary["state_stats"]:
        names = ["pan", "lift", "elbow", "w_flex", "w_roll", "grip"]
        print("\n  关节值范围:")
        print(f"    {'joint':<8} {'state min':>10} {'state max':>10} "
              f"{'action min':>11} {'action max':>11}")
        for i, n in enumerate(names):
            s_min, s_max = summary["state_stats"]["min"][i], summary["state_stats"]["max"][i]
            a_min, a_max = summary["action_stats"]["min"][i], summary["action_stats"]["max"][i]
            print(f"    {n:<8} {s_min:>10.2f} {s_max:>10.2f} {a_min:>11.2f} {a_max:>11.2f}")

    print()
    if summary["issues"]:
        print(f"  ❌ 问题 ({len(summary['issues'])}):")
        for it in summary["issues"]:
            print(f"     • {it}")
    else:
        print("  ✓ 数据完整，无 NaN/Inf，关节维度正确")

    # ---- frame readability quick check ----
    print()
    sample_ep = eps[0]
    sample_frame = sample_ep / "frames" / "frame_00000.jpg"
    if sample_frame.exists():
        import cv2
        img = cv2.imread(str(sample_frame))
        if img is None:
            summary["issues"].append("无法读取首帧 JPG")
            print("  ❌ 无法读取首帧 JPG")
        else:
            print(f"  ✓ 首帧可读: shape={img.shape}, mean={img.mean():.1f} "
                  f"({'有内容' if img.mean() > 5 else '⚠️ 像是黑屏'})")

    return summary


# ---------------------------------------------------------------------------
# Phase 2: SmolVLA load + forward pass
# ---------------------------------------------------------------------------

def smoke_test_smolvla(demos_dir: Path) -> bool:
    print()
    print("=" * 64)
    print("PHASE 2: SmolVLA 加载 + 单步前向")
    print("=" * 64)

    # Try multiple known import paths (lerobot version might differ)
    SmolVLAPolicy = None
    import_attempts = [
        "lerobot.policies.smolvla.modeling_smolvla",
        "lerobot.policies.smolvla.smolvla_policy",
        "lerobot.policies.smolvla",
    ]
    last_err = None
    for path in import_attempts:
        try:
            import importlib
            mod = importlib.import_module(path)
            for attr in ("SmolVLAPolicy", "SmolVLA"):
                if hasattr(mod, attr):
                    SmolVLAPolicy = getattr(mod, attr)
                    print(f"  ✓ 找到 SmolVLA class: {path}.{attr}")
                    break
            if SmolVLAPolicy is not None:
                break
        except Exception as e:
            last_err = e

    if SmolVLAPolicy is None:
        print(f"  ⚠️  无法导入 SmolVLAPolicy。最后错误: {last_err}")
        print("     可能你的 lerobot 版本不带 SmolVLA。可以:")
        print("       pip install --upgrade lerobot")
        print("     或者用 ACT 替代:")
        print("       from lerobot.policies.act.modeling_act import ACTPolicy")
        return False

    # Load pretrained
    print("\n  → 下载/加载 lerobot/smolvla_base 预训练模型 (~1GB首次会比较慢)...")
    try:
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        print(f"  ✓ Pretrained 加载成功")
    except Exception as e:
        print(f"  ❌ from_pretrained 失败: {e}")
        traceback.print_exc()
        return False

    # Inspect features
    cfg = policy.config
    print(f"\n  模型配置:")
    print(f"    input_features:")
    for name, feat in (cfg.input_features or {}).items():
        print(f"      • {name}: shape={getattr(feat, 'shape', '?')}, "
              f"type={type(feat).__name__}")
    print(f"    output_features:")
    for name, feat in (cfg.output_features or {}).items():
        print(f"      • {name}: shape={getattr(feat, 'shape', '?')}")

    # Get expected action dim
    action_feat = cfg.output_features.get("action")
    if action_feat is None:
        print("  ❌ output_features 没有 'action' 键 — 模型 schema 不兼容")
        return False
    expected_dim = action_feat.shape[0] if hasattr(action_feat, "shape") else None
    if expected_dim != 6:
        print(f"  ⚠️  SmolVLA pretrained 期望 action_dim={expected_dim}，"
              f"你的 SO-101 是 6 维。")
        print(f"     SFT 时 lerobot 会重新初始化 action head，所以这不是 blocker。")

    # Build a single observation from the user's first episode
    ep0 = sorted(demos_dir.glob("episode_*"))[0]
    states = np.load(ep0 / "states.npy")
    import cv2, torch
    frame = cv2.imread(str(ep0 / "frames" / "frame_00000.jpg"))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Determine target image size from input_features
    img_feature_names = [n for n in (cfg.input_features or {})
                          if "image" in n.lower() or "rgb" in n.lower()]
    target_h = target_w = 256
    if img_feature_names:
        first_shape = cfg.input_features[img_feature_names[0]].shape
        if len(first_shape) == 3:
            target_h, target_w = int(first_shape[1]), int(first_shape[2])

    rgb_resized = cv2.resize(rgb, (target_w, target_h))
    img_t = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    state_t = torch.from_numpy(states[0:1]).float()
    task_str = ["pick up the carrot"]

    obs = {}
    for name in (cfg.input_features or {}).keys():
        if "image" in name.lower() or "rgb" in name.lower():
            # SO-101 only has one camera — replicate to satisfy multi-cam pretrained
            obs[name] = img_t.clone()
        elif "state" in name.lower():
            obs[name] = state_t

    # Language: tokenize. Try several paths to find the right tokenizer.
    tok = None
    for attr_path in [
        ("model", "vlm_with_expert", "processor", "tokenizer"),
        ("model", "vlm_with_expert", "tokenizer"),
        ("vlm_with_expert", "processor", "tokenizer"),
        ("language_tokenizer",),
    ]:
        obj = policy
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and callable(getattr(obj, "__call__", None)):
            tok = obj
            print(f"\n  ✓ tokenizer found at: policy.{'.'.join(attr_path)}")
            break

    if tok is None:
        # Fallback: load directly from the SmolVLM model id
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            print(f"\n  ✓ tokenizer loaded from HuggingFace AutoTokenizer (fallback)")
        except Exception as e:
            print(f"\n  ❌ AutoTokenizer fallback failed: {e}")

    if tok is not None:
        try:
            toks = tok(task_str, return_tensors="pt", padding=True)
            obs["observation.language.tokens"] = toks["input_ids"].long()
            # attention_mask must be bool for SmolVLA's torch.where(...)
            obs["observation.language.attention_mask"] = toks["attention_mask"].bool()
            print(f"  → input_ids shape={tuple(toks['input_ids'].shape)}, "
                  f"sample tokens={toks['input_ids'][0][:8].tolist()}")
        except Exception as e:
            print(f"  ⚠️ tokenize call failed: {e}")

    print(f"  → 构造 observation: keys={list(obs.keys())}")
    print(f"     image dtype={img_t.dtype} shape={tuple(img_t.shape)} (resized to {target_h}x{target_w})")
    print(f"     state dtype={state_t.dtype} shape={tuple(state_t.shape)}")

    # Forward — first move all tensors onto the policy's device
    device = next(policy.parameters()).device
    print(f"\n  → 移动 obs 到设备 {device} 上 …")
    for k, v in list(obs.items()):
        if hasattr(v, "to"):
            obs[k] = v.to(device)

    print(f"  → 跑 select_action()...")
    try:
        policy.eval()
        policy.reset()
        with torch.no_grad():
            action = policy.select_action(obs)
    except TypeError:
        try:
            with torch.no_grad():
                action = policy.select_action(obs, task_str)
        except Exception as e:
            print(f"  ❌ select_action 失败: {e}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"  ❌ select_action 失败: {e}")
        traceback.print_exc()
        return False

    a = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
    print(f"  ✓ 前向成功! action shape={a.shape}, dtype={a.dtype}")
    print(f"    数值: min={a.min():.2f} max={a.max():.2f} mean={a.mean():.2f}")
    print(f"    样本: {a.flatten()[:6].tolist()}")
    if np.isnan(a).any():
        print("  ❌ action 含 NaN — 模型有问题")
        return False
    return True


# ---------------------------------------------------------------------------
# Optional: trajectory plot
# ---------------------------------------------------------------------------

def plot_episode(demos_dir: Path, ep_idx: int = 0, out_path: str = "/tmp/episode_trajectory.png"):
    print()
    print("=" * 64)
    print(f"PHASE 3 (可选): episode_{ep_idx:04d} 轨迹图")
    print("=" * 64)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = sorted(demos_dir.glob("episode_*"))
    if ep_idx >= len(eps):
        print(f"  ep_idx={ep_idx} 超过总数 {len(eps)}")
        return
    ep = eps[ep_idx]
    states = np.load(ep / "states.npy")
    actions = np.load(ep / "actions.npy")
    fps = json.loads((ep / "meta.json").read_text()).get("fps", 20)
    t = np.arange(states.shape[0]) / fps

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_roll", "gripper"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, states[:, i], label="state", linewidth=1.4)
        ax.plot(t, actions[:, i], label="action (leader)", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_title(names[i])
        ax.set_ylabel("deg")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"{ep.name}  ({states.shape[0]} frames)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  ✓ 轨迹图保存到 {out_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="./demos")
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    demos_dir = Path(args.demos)
    if not demos_dir.exists():
        print(f"❌ {demos_dir} 不存在。先跑 record_demos.py。")
        sys.exit(1)

    summary = check_data(demos_dir)

    if args.plot:
        plot_episode(demos_dir, 0)

    if args.data_only:
        sys.exit(0 if not summary["issues"] else 2)

    ok = smoke_test_smolvla(demos_dir)

    print()
    print("=" * 64)
    if not summary["issues"] and ok:
        print("🎉  全部通过！可以继续录到 ~50 条然后正式训练。")
        sys.exit(0)
    elif not summary["issues"]:
        print("⚠️  数据 OK，但 SmolVLA 加载/前向有问题。看上面错误信息。")
        sys.exit(2)
    else:
        print(f"❌  数据有 {len(summary['issues'])} 处问题，先修。")
        sys.exit(2)


if __name__ == "__main__":
    main()
