import os

os.makedirs("configs", exist_ok=True)
config_content = """compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

with open("configs/accelerate_config.yaml", "w", encoding="utf-8") as f:
    f.write(config_content)

print("[INFO] Created configs/accelerate_config.yaml successfully.")