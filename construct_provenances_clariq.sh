## (1) retrieve K provenances
### Setting: passages indexed using contents and title
### [TODO] indexing using contents only
python3 src/pre/retrieve_passages2.py \
    --clariq data/clariq/train.tsv \
    --output data/clariq_provenances_tc.jsonl \
    --index_dir /tmp2/jhju/indexes/odcqa-psgs \
    --k1 0.9 --b 0.4 \
    --index_dir /home/jhju/indexes/full_wiki_segments_lucene \
    --k 100

### [ALTER] dense indexing
# python3 src/pre/retrieve_passages2.py \
#     --clariq data/clariq/train.tsv \
#     --output data/clariq_provenances_dpr.jsonl \
#     --dense_retrieval \
#     --q-encoder facebook/dpr-question_encoder-multiset-base \
#     --device cuda:2 \
#     --batch_size 32 \
#     --threads 4 \
#     --index_dir /home/jhju/indexes/full_wiki_segments_dpr \
#     --k 100 

## (2) set the provenances for FiD
# python3 src/pre/organize_provenances.py \
#     --questions_with_provenances data/clariq_provenances_dpr.jsonl \
#     --collections /home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl \
#     --output data/train_fidcqg_v0_k20.jsonl \
#     --N 10 \
#     --topk 20 \
#     --overlapped
