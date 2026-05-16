"""
把 demos/ 里的 episode_NNNN/ 转成 LeRobotDataset 格式，给 lerobot 的 train 脚本喂。

输出结构 (训练时直接 dataset.repo_id 指向):
    datasets/local/zhenbang_pickplace/
      data/chunk-000/episode_*.parquet
      videos/chunk-000/observation.images.front/episode_*.mp4
      meta/{info,tasks,episodes}.jsonl

用法:
    python convert_to_lerobot.py
    python convert_to_lerobot.py --input ./demos --repo-id local/zhenbang_pickplace --fps 20
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./demos")
    ap.add_argument("--repo-id", default="local/zhenbang_pickplace")
    ap.add_argument("--root", default="./datasets")
    ap.add_argument("--fps", type=int, default=20)
    # SmolVLA 默认期望 observation.images.camera1 这个键，
    # 用同名能让 pretrained image encoder 权重直接转移
    ap.add_argument("--image-key", default="observation.images.camera1")
    ap.add_argument("--robot-type", default="so101")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的输出目录")
    args = ap.parse_args()

    src = Path(args.input).resolve()
    eps = sorted(src.glob("episode_*"))
    if not eps:
        raise SystemExit(f"❌ {src} 里没有 episode_* 子目录")

    # 读第一帧确定图像尺寸
    sample = cv2.imread(str(eps[0] / "frames" / "frame_00000.jpg"))
    if sample is None:
        raise SystemExit(f"❌ 无法读 {eps[0]/'frames'/'frame_00000.jpg'}")
    h, w = sample.shape[:2]
    print(f"  Image size: {w}x{h}")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        args.image_key: {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
    }

    out_root = Path(args.root).resolve() / args.repo_id
    if out_root.exists():
        if not args.force:
            raise SystemExit(
                f"❌ {out_root} 已存在。加 --force 覆盖，或换 --repo-id")
        print(f"  → 删除旧目录 {out_root}")
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"  → 创建 LeRobotDataset @ {out_root}")
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=out_root,
        robot_type=args.robot_type,
        use_videos=True,
    )

    print(f"\n  → 转换 {len(eps)} 条 episode ...")
    task_counts: dict[str, int] = {}
    total_frames = 0

    for ep_dir in eps:
        meta_path = ep_dir / "meta.json"
        states_path = ep_dir / "states.npy"
        actions_path = ep_dir / "actions.npy"

        if not all(p.exists() for p in (meta_path, states_path, actions_path)):
            print(f"     ⚠ 跳过 {ep_dir.name} (缺文件)")
            continue

        meta = json.loads(meta_path.read_text())
        states = np.load(states_path)
        actions = np.load(actions_path)
        n = states.shape[0]
        instr = meta.get("instruction") or f"do {meta.get('task_name', '?')}"
        task_name = meta.get("task_name", "?")

        if n != actions.shape[0]:
            print(f"     ⚠ {ep_dir.name} states={n} actions={actions.shape[0]} 不一致, 取 min")
            n = min(n, actions.shape[0])

        n_frames_dir = len(list((ep_dir / "frames").glob("*.jpg")))
        if n_frames_dir < n:
            print(f"     ⚠ {ep_dir.name} states={n} but {n_frames_dir} frames, 截到 {n_frames_dir}")
            n = n_frames_dir

        for i in range(n):
            frame_path = ep_dir / "frames" / f"frame_{i:05d}.jpg"
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                print(f"     ⚠ frame {frame_path} 读不到, 跳过")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            ds.add_frame({
                "observation.state": states[i].astype(np.float32),
                args.image_key: rgb,
                "action": actions[i].astype(np.float32),
                "task": instr,
            })
        ds.save_episode()
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
        total_frames += n
        print(f"     ✓ {ep_dir.name}: {n} frames  task={task_name}")

    print(f"\n✓ 完成!")
    print(f"  输出: {out_root}")
    print(f"  总 episodes: {len(eps)}")
    print(f"  总 frames: {total_frames}  (~{total_frames/args.fps:.1f}s = {total_frames/args.fps/60:.1f} min)")
    print(f"  任务分布: {task_counts}")
    print()
    print("  下一步训练命令:")
    print(f"    python -m lerobot.scripts.train \\")
    print(f"      policy=smolvla \\")
    print(f"      policy.pretrained_path=lerobot/smolvla_base \\")
    print(f"      dataset.repo_id={args.repo_id} \\")
    print(f"      dataset.root={out_root} \\")
    print(f"      output_dir=./checkpoints/smolvla_zhenbang")


if __name__ == "__main__":
    main()
