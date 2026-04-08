# 1 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
# 1  GPU   unifiedreward
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:unifiedreward_sd3_1gpu
# 1  GPU   ImageReward
#accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29499 scripts/train_sd3.py --config config/grpo.py:imagereward_sd3
#
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29499 scripts/train_sd3.py --config config/grpo.py:geneval_sd3
## 11

