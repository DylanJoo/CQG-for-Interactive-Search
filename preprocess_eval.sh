CORPUS_DIR=~/datasets/wiki.dump.20181220/
CORPUS=${CORPUS_DIR}/wiki_psgs_w100.jsonl
DATASET_DIR=/home/jhju/huggingface_hub/CQG/

# 3-1
# retrieve_serp_qrecc_bm25:
# python3 src/data_augmentation/retrieve_passages.py \
#     --qrecc data/qrecc/qrecc_test.json \
#     --output ${DATASET_DIR}/qrecc_test_provenances_bm25.jsonl \
#     --k1 0.9 --b 0.4 \
#     --index_dir /home/jhju/indexes/wikipedia-lucene \
#     --k 100

# retrieve_serp_qrecc_contriever:
python3 src/data_augmentation/retrieve_passages.py \
    --qrecc data/qrecc/qrecc_test.json \
    --output ${DATASET_DIR}/qrecc_test_provenances_contriever.jsonl \
    --dense_retrieval \
    --q-encoder facebook/contriever-msmarco \
    --device cuda:2 \
    --batch_size 54 \
    --threads 4 \
    --index_dir /home/jhju/indexes/wikipedia-contriever \
    --k 100 
