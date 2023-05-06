## (1) retrieve K provenances
### Lucene index using contents with title (title content)
# python3 src/pre/retrieve_passages2.py \
#     --clariq data/clariq/train.tsv \
#     --output data/clariq_provenances_lucene.jsonl \
#     --k1 0.9 --b 0.4 \
#     --index_dir /home/jhju/indexes/full_wiki_segments_lucene \
#     --k 100

### FAISS index using contents with title (title [SEP] content)
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
for indexing in lucene dpr;do
    python3 src/pre/organize_provenances.py \
        --questions_with_provenances data/clariq_provenances_${indexing}.jsonl \
        --collections /home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl \
        --output data/train_fidcqg_v0_${indexing}.jsonl \
        --N 10 \
        --overlapped
done
