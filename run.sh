# Run the results 
python3 src/inference_fidcqg.py \
    --jsonl_file data/canard_fidcqg.jsonl \
    --output_file data/canard_cquestions.jsonl \
    --batch_size 4 \
    --used_checkpoint fidcqg/checkpoint-10000/ \
    --used_tokenizer t5-base \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 256 \
    --device 'cuda'
