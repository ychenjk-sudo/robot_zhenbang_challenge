"""
训练 loss 曲线监控。
读 /tmp/train.log，解析 lerobot 训练步骤行，更新 PNG 曲线 + 终端摘要。

用法:
    python monitor_training.py                       # 默认每 30s 刷新
    python monitor_training.py --log /tmp/train.log --interval 60
    python monitor_training.py --once                # 只画一次
"""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 匹配两类 lerobot 训练日志:
#  • 进度条:    "Training:  25%| | 1530/6000 [...]"  ← 真实 step 数
#  • INFO 行:   "step:1K smpl:3K ... loss:0.371 grdn:7.4 lr:8.5e-05"  ← K 缩写, 没真值
# 我们用进度条匹配 step + 时间戳, 然后从最近的 INFO 行拿 loss/grdn/lr.
PROGRESS_RE = re.compile(r"(?P<step>\d+)/\d+\s*\[")
INFO_RE = re.compile(
    r"loss:(?P<loss>[\d.]+).*?"
    r"grdn:(?P<grdn>[\d.]+).*?"
    r"lr:(?P<lr>[\d.eE+-]+)"
)


def parse_log(path: Path):
    """
    Stream through the log:
      • Track the latest step seen via progress-bar lines.
      • When an INFO line with loss appears, pair it with the most recent step.
    """
    if not path.exists():
        return []
    rows = []
    last_step: int = 0
    text = path.read_text(errors="ignore")
    # The progress bar uses \r overwrites — split on \r AND \n
    for line in re.split(r"[\r\n]+", text):
        for pm in PROGRESS_RE.finditer(line):
            try:
                last_step = int(pm["step"])
            except ValueError:
                pass
        im = INFO_RE.search(line)
        if im and last_step > 0:
            try:
                rows.append((
                    last_step,
                    float(im["loss"]),
                    float(im["grdn"]),
                    float(im["lr"]),
                ))
            except ValueError:
                continue
    # Deduplicate consecutive same-step entries (keep last)
    out, seen = [], set()
    for s, l, g, lr in reversed(rows):
        if s not in seen:
            seen.add(s)
            out.append((s, l, g, lr))
    return list(reversed(out))


def render_plot(rows, out_path: str, target_steps: int = 6000):
    if not rows:
        return None
    arr = np.array(rows)
    steps = arr[:, 0]
    loss  = arr[:, 1]
    grdn  = arr[:, 2]
    lr    = arr[:, 3]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax = axes[0]
    ax.plot(steps, loss, "o-", color="#1f77b4", linewidth=1.6, markersize=3)
    ax.set_ylabel("loss")
    ax.set_title(f"SmolVLA SFT — step {int(steps[-1])} / {target_steps}  "
                 f"current loss={loss[-1]:.3f}")
    ax.grid(alpha=0.3)

    # Reference bands
    for y, label, color in [
        (0.20, "usable (0.20)",       "#ff7f0e"),
        (0.15, "well-trained (0.15)", "#2ca02c"),
        (0.10, "excellent (0.10)",    "#9467bd"),
    ]:
        ax.axhline(y, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(steps[-1] * 0.98, y, label, ha="right", va="bottom",
                color=color, fontsize=9, alpha=0.8)
    ax.set_ylim(0, max(0.3, loss.max() * 1.05))

    ax2 = axes[1]
    ax2.plot(steps, grdn, "o-", color="#d62728", linewidth=1.4, markersize=2)
    ax2.set_ylabel("gradient norm")
    ax2.axhline(5.0, linestyle="--", alpha=0.4, color="green",
                label="healthy (<5)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(steps, lr, "o-", color="#2ca02c", linewidth=1.4, markersize=2)
    ax3.set_ylabel("learning rate")
    ax3.set_xlabel("step")
    ax3.set_yscale("log")
    ax3.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return arr


def print_summary(arr):
    if arr is None or len(arr) == 0:
        print("    (尚无 loss 数据)")
        return
    last = arr[-1]
    first_loss = arr[0, 1]
    last_loss = last[1]
    drop_ratio = last_loss / first_loss if first_loss > 0 else 1.0

    # Health status
    if last_loss < 0.10:
        health = "🎉 优秀"
    elif last_loss < 0.15:
        health = "✅ 充分训练"
    elif last_loss < 0.20:
        health = "✅ 可用"
    elif last_loss < 0.40:
        health = "🟡 学习中"
    else:
        health = "🔴 太高 / 没学进去"

    # Recent slope (last 10 points)
    if len(arr) >= 10:
        recent = arr[-10:]
        slope = np.polyfit(recent[:, 0], recent[:, 1], 1)[0]
        slope_str = f"{slope:+.2e} / step"
        if abs(slope) < 1e-5:
            slope_status = "🟡 趋于平稳，接近收敛"
        elif slope < 0:
            slope_status = "📉 还在下降"
        else:
            slope_status = "⚠️  最近在上升（异常）"
    else:
        slope_str = "(数据点太少)"
        slope_status = ""

    print(f"  Step:        {int(last[0])}")
    print(f"  Loss:        {last_loss:.4f}  ({health})")
    print(f"  Drop:        {first_loss:.3f} → {last_loss:.3f}  "
          f"(比例 {drop_ratio:.2%})")
    print(f"  Gradient:    {last[2]:.3f}")
    print(f"  LR:          {last[3]:.2e}")
    print(f"  最近斜率:     {slope_str}  {slope_status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/train.log")
    ap.add_argument("--out", default="/tmp/loss_curve.png")
    ap.add_argument("--interval", type=int, default=30, help="刷新秒数")
    ap.add_argument("--target-steps", type=int, default=6000)
    ap.add_argument("--once", action="store_true", help="只画一次就退出")
    args = ap.parse_args()

    log = Path(args.log)
    print(f"Monitor → {log}")
    print(f"Curve  → {args.out}")
    if not args.once:
        print(f"刷新间隔: {args.interval}s   (Ctrl+C 退出)")

    while True:
        rows = parse_log(log)
        arr = render_plot(rows, args.out, target_steps=args.target_steps)
        ts = time.strftime("%H:%M:%S")
        print(f"\n──── {ts} ────")
        print_summary(arr)
        print(f"  曲线: {args.out}")
        if args.once:
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n退出。")
            break


if __name__ == "__main__":
    main()
