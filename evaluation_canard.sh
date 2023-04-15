mkdir -p evaluation

## (1) retrieve K provenances
# python3 src/pre/retrieve_passages.py \
#     --canard data/canard/test.jsonl \
#     --collections None \
#     --output evaluation/data/canard_provenances_bm25.jsonl \
#     --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc/ \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4

## (2.1) set the provenances for FiD
## [NOTE] Here we can also adopt another retrieval model 
## to update the provenances.
# python3 src/pre/organize_provenances.py \
#     --questions_with_provenances evaluation/canard_provenances_bm25.jsonl \
#     --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
#     --output evaluation/data/canard_qa_provenances.jsonl \
#     --N 10
## (2.2) set the answer
# python3 src/pre/organize_miresponses.py \
#     --convqa evaluation/canard_qa_provenances.jsonl \
#     --output evaluation/canard_test_fidmrg_v0.jsonl

## (3) Generate clarification questions
python3 src/inference_fidmrg.py \
    --jsonl_file evaluation/canard_test_fidmrg_v0.jsonl \
    --output_file evaluation/canard_test_pred.jsonl \
    --batch_size 2 \
    --used_checkpoint fidmrg/checkpoint-10000/ \
    --used_tokenizer t5-base \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 256 \
    --device 'cuda'
