

accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29499 scripts/train_sd3_cpo2.py --config config/cpo.py:geneval_sd3

## 11

