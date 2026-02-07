#!/bin/bash
# === WandB 在线配置 ===
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=online   # 开启在线同步
export MASTER_PORT=29505

# === 显存保护 ===
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# === 训练参数 ===
# 目标：跑满 5000 步，每 1000 步存一个档
# 显存策略：BS=2, Accum=8 (等效 BS=16)

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/root/autodl-tmp/openvla/checkpoints/openvla-7b" \
  --data_root_dir "/root/autodl-tmp/datasets" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "/root/autodl-tmp/openvla-finetuned" \
  --adapter_tmp_dir "/root/autodl-tmp/adapter_tmp" \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 1000 \
  --max_steps 5000 \
  --wandb_project "" \
  --wandb_entity ""  

