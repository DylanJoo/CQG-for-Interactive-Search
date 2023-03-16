# setting 1
## - title+content as the retrieval of provenances
## - content as the source of provenances
python3 src/pre/organize_provenances.py \
    --clariq_provenances data/clariq_provenances_tc.jsonl \
    --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
    --output data/train_cqg_v0.jsonl \
    --N 10
