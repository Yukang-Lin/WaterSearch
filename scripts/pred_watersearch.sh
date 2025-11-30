export CUDA_VISIBLE_DEVICES=0

python pred_chunk_rougeL.py \
  --mode old \
  --gamma 0.45 \
  --delta 5 \
  --bl_type hard \
  --model qwen2.5-7b \
  --dataset all \
  --K 4 \
  --alpha 0.75 \
  --custom_text "search"

# python pred_sentence_rougeL.py \
#   --mode old \
#   --gamma 0.45 \
#   --delta 5 \
#   --bl_type hard \
#   --model qwen2.5-7b \
#   --dataset all \
#   --K 4 \
#   --alpha 0.75 \
#   --custom_text "search"


# python pred_chunk_semsim.py \
#   --mode old \
#   --gamma 0.45 \
#   --delta 5 \
#   --bl_type hard \
#   --model qwen2.5-7b \
#   --dataset all \
#   --K 4 \
#   --alpha 0.75 \
#   --custom_text "search"