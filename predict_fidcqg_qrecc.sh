python3 src/inference_fidcqg.py \
    --jsonl_file data/qrecc_provenances_bm25.jsonl \
    --output_file evaluation/cqg/dev_${model}-${ckpt}-${version}_pred.jsonl \
    --batch_size 4 \
    --used_checkpoint checkpoints/fidcqg.bm25.ovl/checkpoint-4000 \
    --used_tokenizer flan-t5-base \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 64 \
    --device cuda:2 \
    --do_sample \
    --top_k 10
# python3 src/tools/convert_jsonl_to_txt.py \
#     --path_jsonl evaluation/cqg/dev_${model}-${ckpt}-${version}_pred.jsonl \
#     --path_txt evaluation/cqg/dev_${model}-${version}_pred.txt
