export CUDA_VISIBLE_DEVICES=0

python detect_chunk.py \
    --input_dir pred/qwen2.5-7b_old_g0.45_d5.0_alpha0.75_hard_search \
    --gamma 0.45 \
    --delta 5.0 \
    --beam_num 4 \
    --chunk_size 20 \
    --mode old \
    --threshold 0.01 \
    --test_min_tokens 6

# python detect_sentence.py \
#     --input_dir pred/qwen2.5-7b_old_g0.45_d5.0_alpha0.75_hard_search \
#     --gamma 0.45 \
#     --delta 5.0 \
#     --beam_num 4 \
#     --chunk_size 20 \
#     --mode old \
#     --threshold 0.01 \
#     --test_min_tokens 6
