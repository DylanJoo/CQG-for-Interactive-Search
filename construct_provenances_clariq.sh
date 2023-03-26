## (1) retrieve K provenances

### Setting 1: passages indexed by contents
# python3 src/pre/retrieve_passages.py \
#     --clariq data/clariq/train.tsv \
#     --collections None \
#     --output data/clariq_provenances.jsonl \
#     --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_c/ \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4

### Setting 2: passages indexed by (contents + title)
# python3 src/pre/retrieve_passages.py \
#     --clariq data/clariq/train.tsv \
#     --collections None \
#     --output data/clariq_provenances_tc.jsonl \
#     --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc/ \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4

## (2) set the provenances for FiD
python3 src/pre/organize_provenances.py \
    --questions_with_provenances data/clariq_provenances_tc.jsonl \
    --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
    --output data/train_fidcqg_v0.jsonl \
    --N 10
