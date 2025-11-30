export CUDA_VISIBLE_DEVICES=0

python pred_baseline.py \
    --mode old \
    --gamma 0.35 \
    --delta 5 \
    --bl_type hard \
    --model qwen2.5-7b \
    --dataset all \
    --custom_text kgw