python3 src/inference_fidcqg.py \
    --jsonl_file data/qrecc_provenances_bm25.jsonl \
    --output_file predictions/qrecc_cq_pred.fidcqg.bm25.ovl.jsonl \
    --collections ~/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl\
    --batch_size 6 \
    --used_checkpoint checkpoints/fidcqg.bm25.ovl/checkpoint-4000 \
    --used_tokenizer google/flan-t5-base \
    --calculate_crossattention  \
    --n_contexts 10 \
    --batch_size 8 \
    --max_length 64 \
    --device cuda \
    --num_beams 5
