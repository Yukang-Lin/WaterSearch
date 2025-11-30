export CUDA_VISIBLE_DEVICES=0

python detect_baseline.py \
    --input_dir pred/qwen2.5-7b_old_g0.35_d5.0_kgw \
    --threshold 4