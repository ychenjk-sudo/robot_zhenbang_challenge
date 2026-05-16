#!/bin/bash
# SmolVLA SFT 训练脚本 — Mac MPS 配置
#
# 用法:
#   ./train.sh smoke    # 跑 100 步验证流程（~10 分钟）
#   ./train.sh full     # 完整训练 6000 步
#   ./train.sh resume   # 从 last checkpoint 续训到 step 6000

set -e

PYTHON=/Users/chenyuying/miniconda3/envs/zhenbang/bin/python
LEROBOT_REPO=/Users/chenyuying/Downloads/lerobot_repo
DATASET_ROOT=/Users/chenyuying/Downloads/lerobot/challenge_pkg/robot_zhenbang_challenge/datasets/local/zhenbang_pickplace
OUTPUT_DIR=./checkpoints/smolvla_zhenbang
JOB_NAME=zhenbang_pickplace

MODE=${1:-smoke}

# Mode-specific overrides
case $MODE in
  smoke)
    STEPS=100
    SAVE_FREQ=100
    LOG_FREQ=10
    OUTPUT_DIR=./checkpoints/smolvla_zhenbang_smoke
    echo "▶ SMOKE 训练 ($STEPS 步) — 验证流程不 crash"
    ;;
  full)
    STEPS=6000
    SAVE_FREQ=3000
    LOG_FREQ=50
    echo "▶ FULL 训练 ($STEPS 步) — Mac MPS 预计 ~6-8 小时"
    ;;
  long)
    STEPS=10000
    SAVE_FREQ=1000
    LOG_FREQ=50
    echo "▶ LONG 训练 ($STEPS 步) — Mac MPS 预计 ~14 小时"
    ;;
  resume)
    STEPS=6000
    SAVE_FREQ=3000
    LOG_FREQ=50
    RESUME_CONFIG=/Users/chenyuying/Downloads/lerobot_repo/checkpoints/smolvla_zhenbang/checkpoints/last/pretrained_model/train_config.json
    if [ ! -f "$RESUME_CONFIG" ]; then
      echo "✗ 找不到 $RESUME_CONFIG"; exit 1
    fi
    echo "▶ 续训模式 (从 last checkpoint 继续到 step $STEPS)"
    ;;
  *)
    echo "未知模式: $MODE  (smoke|full|resume|long)"
    exit 1
    ;;
esac

mkdir -p $OUTPUT_DIR

cd $LEROBOT_REPO

if [ "$MODE" == "resume" ]; then
  # Resume: must NOT pass --policy.path (it would override resume).
  # config_path tells lerobot where the saved config is, and it auto-derives
  # checkpoint_path from there.
  $PYTHON -m lerobot.scripts.lerobot_train \
      --config_path=$RESUME_CONFIG \
      --resume=true \
      --steps=$STEPS \
      --save_freq=$SAVE_FREQ \
      --log_freq=$LOG_FREQ \
      --num_workers=0 \
      --wandb.enable=false
else
  $PYTHON -m lerobot.scripts.lerobot_train \
      --policy.path=lerobot/smolvla_base \
      --policy.push_to_hub=false \
      --policy.device=mps \
      --policy.optimizer_lr=1e-4 \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --policy.num_vlm_layers=16 \
      --dataset.repo_id=local/zhenbang_pickplace \
      --dataset.root=$DATASET_ROOT \
      --batch_size=2 \
      --steps=$STEPS \
      --save_freq=$SAVE_FREQ \
      --log_freq=$LOG_FREQ \
      --num_workers=0 \
      --output_dir=$OUTPUT_DIR \
      --job_name=$JOB_NAME \
      --wandb.enable=false
fi
