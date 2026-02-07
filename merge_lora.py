import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import os
import shutil

# === 配置路径 ===
base_model_path = "/root/autodl-tmp/openvla/checkpoints/openvla-7b"
adapter_root = "/root/autodl-tmp/openvla-finetuned"
all_subdirs = [os.path.join(adapter_root, d) for d in os.listdir(adapter_root) if os.path.isdir(os.path.join(adapter_root, d))]
adapter_path = max(all_subdirs, key=os.path.getmtime) 

output_path = "/root/autodl-tmp/openvla-libero-final"

print(f"Base Model: {base_model_path}")
print(f"Adapter: {adapter_path}")

# 加载并合并
print("Loading and merging...")
base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)
merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload()

# 保存
print(f"Saving to {output_path}...")
merged_model.save_pretrained(output_path)
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.save_pretrained(output_path)
print("Merge Complete!")
