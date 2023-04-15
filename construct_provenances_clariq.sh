## (1) retrieve K provenances
### Setting: passages indexed using contents and title
### [ALTER] indexing using contents only
python3 src/pre/retrieve_passages.py \
    --clariq data/clariq/train.tsv \
    --collections None \
    --output data/clariq_provenances_tc.jsonl \
    --index_dir /tmp2/jhju/indexes/odcqa-psgs \
    --k 100 \
    --k1 0.9 \
    --b 0.4

## (2) set the provenances for FiD
python3 src/pre/organize_provenances.py \
    --questions_with_provenances data/clariq_provenances_tc.jsonl \
    --collections /tmp2/jhju/datasets/odcqa-psgs/full_wiki_segments.jsonl \
    --output data/train_fidcqg_v0.jsonl \
    --N 10 \
    --overlapped
