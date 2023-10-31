COLLECTION=~/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl
MODEL=checkpoints/fidmrg_random.bm25.fidcqg-w/checkpoint-10000

for MODEL in checkpoints/fidmrg*;do
    python3 src/inference_fidmrg.py \
        --jsonl_file data/qrecc_provenances_bm25.jsonl \
        --output_file evaluation/qrecc_train_pred.fidmrg_random.bm25.fidcqg-w.jsonl \
        --collections $COLLECTION \
        --used_checkpoint $MODEL \
        --used_tokenizer google/flan-t5-base \
        --calculate_crossattention \
        --n_contexts 10 \
        --batch_size 4 \
        --max_length 64 \
        --device cuda:1 \
        --num_beams 1
done
