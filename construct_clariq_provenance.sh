# contents
# python3 src/pre/retrieve_provenance.py \
#     --clariq data/clariq/train.tsv \
#     --collections None \
#     --output data/clariq_provenances.jsonl \
#     --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_c/ \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4

# contents + title
python3 src/pre/retrieve_provenance.py \
    --clariq data/clariq/train.tsv \
    --collections None \
    --output data/clariq_provenances_tc.jsonl \
    --index_dir /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc/ \
    --k 100 \
    --k1 0.9 \
    --b 0.4
