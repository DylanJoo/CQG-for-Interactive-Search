COLLECTION=/home/jhju/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl

for MODEL in checkpoints/fidmrg*;do
    MODEL_NAME=${MODEL##*/}
    python3 src/inference_fidmrg.py \
        --jsonl_file data/is4qrecc/qrecc_test_provenances_bm25.jsonl \
        --output_file evaluation/qrecc_test_pred_bm25.$MODEL_NAME.jsonl \
        --collections $COLLECTION \
        --used_checkpoint $MODEL/checkpoint-10000 \
        --used_tokenizer google/flan-t5-base \
        --calculate_crossattention \
        --n_contexts 10 \
        --batch_size 4 \
        --max_length 64 \
        --device cuda:1 \
        --num_beams 1
done
