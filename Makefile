INDEX_DIR=/home/jhju/indexes/wikipedia-lucene
CORPUS_DIR=~/datasets/wiki.dump.20181220/
CORPUS=${CORPUS_DIR}/wiki_psgs_w100.jsonl
DATASET_DIR=/home/jhju/huggingface_hub/CQG/

preprocess_wiki_corpus:
	python3 src/tools/tsv_to_jsonl.py \
	  --path_tsv ${CORPUS_DIR}/wiki_psgs_w100.tsv \
	  --path_jsonl ${CORPUS_DIR}/wiki_psgs_w100.jsonl
# 1-1
index_cc_bm25:
	python3 -m pyserini.index.lucene \
	  --collection JsonCollection \
	  --input /home/jhju/datasets/qrecc/collection-paragraph/ \
	  --index /home/jhju/indexes/qrecc-commoncrawl-lucene/  \
	  --generator DefaultLuceneDocumentGenerator \
	  --threads 8

index_wiki_corpus_bm25:
	python3 -m pyserini.index.lucene \
	  --collection JsonCollection \
	  --input ${CORPUS_DIR} \
	  --index /home/jhju/indexes/wikipedia-lucene \
	  --generator DefaultLuceneDocumentGenerator \
	  --fields title \
	  --threads 8

# 1-2 (sparse)
retrieve_serp_clariq_bm25:
	python3 src/data_augmentation/retrieve_passages.py \
	    --clariq data/clariq/train.tsv \
	    --output data/clariq_provenances_bm25.jsonl \
	    --k1 0.9 --b 0.4 \
	    --index_dir /home/jhju/indexes/wikipedia-lucene \
	    --k 100

# 1-2 (dense)
retrieve_serp_clariq_contriever:
	python3 src/data_augmentation/retrieve_passages.py \
	    --clariq data/clariq/train.tsv \
	    --output data/clariq_provenances_contriever.jsonl \
	    --dense_retrieval \
	    --q-encoder facebook/contriever-msmarco \
	    --device cuda:2 \
	    --batch_size 32 \
	    --threads 4 \
	    --index_dir /home/jhju/indexes/wikipedia-contriever \
	    --k 100 

# 1-3
construct_provenances_fidcqg:
	python3 src/data_augmentation/handler.py \
	    --input ${DATASET_DIR}/clariq_provenances_bm25.jsonl \
	    --output ${DATASET_DIR}/fidcqg.train.bm25.ovl.jsonl \
	    --collections ${CORPUS} \
	    --topk 100 --N 10 --overlapped
	python3 src/data_augmentation/handler.py \
	    --input ${DATASET_DIR}/clariq_provenances_contriever.jsonl \
	    --output ${DATASET_DIR}/fidcqg.train.contriever.ovl.jsonl \
	    --collections ${CORPUS} \
	    --topk 100 --N 10 --overlapped
# 2
train_fidcqg_naive:
	echo "train_fidcqg_naive.sh"

train_fidcqg_weighted:
	echo "train_fidcqg_weighted.sh"


# 3-1
retrieve_serp_qrecc_bm25:
	python3 src/data_augmentation/retrieve_passages.py \
	    --qrecc data/qrecc/qrecc_train.json \
	    --output ${DATASET_DIR}/qrecc_provenances_bm25.jsonl \
	    --k1 0.9 --b 0.4 \
	    --index_dir /home/jhju/indexes/wikipedia-lucene \
	    --k 100

retrieve_serp_qrecc_contriever:
	python3 src/data_augmentation/retrieve_passages.py \
	    --qrecc data/qrecc/qrecc_train.json \
	    --output ${DATSET_DIR}/qrecc_provenances_contriever.jsonl \
	    --dense_retrieval \
	    --q-encoder facebook/contriever-msmarco \
	    --device cuda:2 \
	    --batch_size 54 \
	    --threads 4 \
	    --index_dir /home/jhju/indexes/wikipedia-contriever \
	    --k 100 

# 3-2
## use q_serp instead of ref_serp. 
## Other type of SERP should prbbly be considered.
predict_provenances_qrecc: 
	python3 src/inference_fidcqg.py \
	    --jsonl_file data/qrecc_provenances_bm25.jsonl \
	    --output_file predictions/qrecc_cq_pred.fidcqg.bm25.ovl.jsonl \
	    --collections ~/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl\
	    --batch_size 6 \
	    --used_checkpoint checkpoints/fidcqg.bm25.ovl/checkpoint-4000 \
	    --used_tokenizer google/flan-t5-base \
	    --calculate_crossattention  \
	    --n_contexts 10 \
	    --batch_size 8 \
	    --max_length 64 \
	    --device cuda \
	    --num_beams 5

# 3-3
PRED_CQ=predictions/qrecc_cq_pred.fidcqg.bm25.ovl.jsonl
PRED_CQ=predictions/qrecc_cq_pred.fidcqg-w.bm25.ovl.jsonl
construct_provenances_qrecc:
	# overlapped
	python3 src/data_augmentation/handler.py \
	    --input ${DATASET_DIR}/qrecc_provenances_bm25.jsonl \
	    --input_cqg_predictions ${PRED_CQ} \
	    --output ${DATASET_DIR}/fidmrg.train.bm25.w.ovl_cqpred.bm25.ovl.jsonl \
	    --collections ${CORPUS} \
	    --topk 100 --N 10 --overlapped
