## (1) retrieve K provenances
# python3 src/pre/retrieve_passages.py \
#     --canard data/canard/train.jsonl \
#     --collections None \
#     --output data/canard_provenances_tc.jsonl \
#     --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc/ \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4

## (2) set the provenances for FiD
# python3 src/pre/organize_provenances.py \
#     --questions_with_provenances data/canard_provenances_tc.jsonl \
#     --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
#     --output data/test_canard_fidcqg.jsonl \
#     --N 10

## (3) Generate clarification questions
python3 src/inference_fidcqg.py \
    --jsonl_file data/test_canard_fidcqg.jsonl \
    --output_file data/canard_provenances_tc_cq.jsonl \
    --batch_size 4 \
    --used_checkpoint fidcqg/checkpoint-10000/ \
    --used_tokenizer t5-base \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 256 \
    --device 'cuda'

## (4) Merge the 'predicted' clarification questions
