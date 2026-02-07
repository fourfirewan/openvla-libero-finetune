import os
from huggingface_hub import snapshot_download

# 设置下载目录
save_dir = "/root/autodl-tmp/datasets"
os.makedirs(save_dir, exist_ok=True)

print("开始下载 ")
print("数据源: openvla/modified_libero_rlds")

# 执行下载 
try:
    snapshot_download(
        repo_id="openvla/modified_libero_rlds", 
        repo_type="dataset",
        local_dir=save_dir,
        local_dir_use_symlinks=False,  # 关键：防止生成软链接
        resume_download=True
    )
    print("下载成功！")
except Exception as e:
    print(f"下载失败: {e}")
    print("尝试备用源: real-lab/libero_filtered_noops_rlds_dataset")
    # 备用方案 (如果上面的挂了)
    snapshot_download(
        repo_id="real-lab/libero_filtered_noops_rlds_dataset", 
        repo_type="dataset",
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
