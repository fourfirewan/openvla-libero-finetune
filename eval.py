import sys
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
import draccus
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, "/root/autodl-tmp/openvla")


try:
    from libero import benchmark
except ImportError:
    from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

@dataclass
class GenerateConfig:
    pretrained_checkpoint: str
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 1
    local_log_dir: str = "./demo_videos"
    model_family: str = "openvla"
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 7
    unnorm_key: Optional[str] = None 

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    print(f"Loading Model from: {cfg.pretrained_checkpoint}")
    set_seed_everywhere(cfg.seed)
    
    # 加载统计数据
    stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if not os.path.exists(stats_path):
        print(f"CRITICAL ERROR: Stats file not found at {stats_path}")
        return
    with open(stats_path, 'r') as f:
        stats_data = json.load(f)
    available_keys = list(stats_data.keys())
    target_key = next((k for k in available_keys if "no_noops" in k), available_keys[0])
    ACTION_MEAN = np.array(stats_data[target_key]['action']['mean'])
    ACTION_STD = np.array(stats_data[target_key]['action']['std'])
    print(f"Loaded Stats Backup using key: {target_key}")

    # 初始化
    model = get_model(cfg)
    if not hasattr(model, 'norm_stats'):
        model.norm_stats = {}
    model.norm_stats[target_key] = stats_data[target_key]
    cfg.unnorm_key = target_key
    processor = get_processor(cfg)
    
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    resize_size = get_image_resize_size(cfg)
    
    task_id = 0
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    
    print(f"\n Start Evaluating Task: {task_description}")

    # === 计分板 ===
    success_count = 0

    for episode_idx in range(cfg.num_trials_per_task):
        env.reset()
        # 确保 idx 不超过初始状态列表长度 (循环使用)
        state_idx = episode_idx % len(initial_states)
        obs = env.set_init_state(initial_states[state_idx])
        
        t = 0
        replay_images = []
        max_steps = 300
        done = False
        
        while t < max_steps + 10:
            if t < 10:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            img = get_libero_image(obs, resize_size)
            replay_images.append(img)
            
            observation = {
                "full_image": img,
                "state": np.concatenate((obs["robot0_eef_pos"], 
                                       quat2axisangle(obs["robot0_eef_quat"]), 
                                       obs["robot0_gripper_qpos"]))
            }

            try:
                action = get_action(cfg, model, observation, task_description, processor=processor)
            except Exception as e:
                print(f"Error: {e}")
                break

            # === 手动修复逻辑 ===
            raw_action_move = action[:3]
            if np.max(np.abs(raw_action_move)) < 0.05:
                # 仅在第一次触发时打印警告，避免刷屏
                if t == 11 and episode_idx == 0:
                    print(f"FORCE APPLYING UN-NORM ACTIVATED.")
                action = action * ACTION_STD + ACTION_MEAN
                if t < 30: action[4] -= 0.05

            action = normalize_gripper_action(action, binarize=True)
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)
            
            obs, _, done, _ = env.step(action.tolist())
            if done: break
            t += 1

        # 保存视频
        save_rollout_video(replay_images, episode_idx, success=done, task_description=task_description, log_file=None)
        
        # 统计结果
        if done:
            success_count += 1
            print(f"Trial {episode_idx+1}/{cfg.num_trials_per_task}: Success")
        else:
            print(f"Trial {episode_idx+1}/{cfg.num_trials_per_task}: Failed")

    # === 最终成绩单 ===
    success_rate = success_count / cfg.num_trials_per_task
    print("\n" + "="*30)
    print(f"Final Evaluation Report")
    print(f"Task: {task_description}")
    print(f"Total Trials: {cfg.num_trials_per_task}")
    print(f"Successes: {success_count}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    eval_libero()
