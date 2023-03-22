## (1) retrieve K provenances
# python3 src/pre/retrieve_passages.py \
#     --clariq data/clariq/train.tsv \
#     --collections None \
#     --output data/clariq_provenances_tc.jsonl \
#     --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc/ \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4

## (2) Generate clarification questions
python3 src/inference_fidcqg.py \
    --jsonl_file data/train_cqg_v0.sample.eval.jsonl \
    --output_file removeme.txt \
    --used_checkpoint fidcqg/checkpoint-10000/ \
    --used_tokenizer t5-small \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 50 \
    --device 'cuda' \

## (3) set the provenances for FiD
# python3 src/pre/organize_provenances.py \
#     --clariq_provenances data/clariq_provenances_tc.jsonl \
#     --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
#     --output data/train_cqg_v0.jsonl \
#     --N 10

