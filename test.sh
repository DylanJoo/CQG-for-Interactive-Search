python3 src/inference_fidcqg.py \
    --jsonl_file data/train_fidcqg_v0.jsonl \
    --output_file data/train_fidcqg_pred.txt \
    --batch_size 2 \
    --used_checkpoint fidcqg/checkpoint-10000/ \
    --used_tokenizer t5-base \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 256 \
    --device 'cuda'
