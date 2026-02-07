#!/bin/bash
# === 显存保护 ===
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# === EGL 驱动注入 ===
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# === Libero 路径 ===
export PYTHONPATH=$PYTHONPATH:/root/autodl-tmp/LIBERO
export LIBERO_DATA_DIR=/root/autodl-tmp/libero_data

# === 模型路径 ===
MODEL_PATH="/root/autodl-tmp/openvla-libero-final"

echo "Starting Batch Evaluation (20 Episodes)..."
echo "Model: $MODEL_PATH"

# 运行评估脚本
xvfb-run -a python eval.py \
  --pretrained_checkpoint "$MODEL_PATH" \
  --task_suite_name "libero_spatial" \
  --num_trials_per_task 20 \
  --local_log_dir "./demo_videos"
