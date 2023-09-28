CORPUS_DIR=~/datasets/wiki.dump.20181220/
CORPUS=${CORPUS_DIR}/wiki_psgs_w100.jsonl
DATASET_DIR=/home/jhju/huggingface_hub/CQG/

preprocess_wiki_corpus:
	python3 src/tools/tsv_to_jsonl.py \
	  --path_tsv ${CORPUS_DIR}/wiki_psgs_w100.tsv \
	  --path_jsonl ${CORPUS_DIR}/wiki_psgs_w100.jsonl
# 1-1
index_wiki_corpus_bm25:
	python3 -m pyserini.index.lucene \
	  --collection JsonCollection \
	  --input ${CORPUS_DIR} \
	  --index ${INDEX_DIR} \
	  --generator DefaultLuceneDocumentGenerator \
	  --fields title \
	  --threads 8

# 1-2a
# retrieve_serp_clariq:
# 	python3 src/data_augmentation/retrieve_passages.py \
# 	    --clariq data/clariq/train.tsv \
# 	    --output data/clariq_provenances_bm25.jsonl \
# 	    --k1 0.9 --b 0.4 \
# 	    --index_dir ${INDEX_DIR} \
# 	    --k 100
#
# retrieve_serp_clariq_contriever:
# 	python3 src/data_augmentation/retrieve_passages.py \
# 	    --clariq data/clariq/train.tsv \
# 	    --output data/clariq_provenances_contriever.jsonl \
# 	    --dense_retrieval \
# 	    --q-encoder facebook/contriever-msmarco \
# 	    --device cuda:2 \
# 	    --batch_size 32 \
# 	    --threads 4 \
# 	    --index_dir ${INDEX_DIR} \
# 	    --k 100 

# 1-3a
construct_provenances_clariq:
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

# 1-2b
# retrieve_serp_qrecc:
# 	python3 src/data_augmentation/retrieve_passages.py \
# 	    --qrecc data/qrecc/qrecc_train.json \
# 	    --output ${DATASET_DIR}/qrecc_provenances_bm25.jsonl \
# 	    --k1 0.9 --b 0.4 \
# 	    --index_dir ${INDEX_DIR} \
# 	    --k 100
#
# retrieve_serp_qrecc_contriever:
# 	python3 src/data_augmentation/retrieve_passages.py \
# 	    --qrecc data/qrecc/qrecc_train.json \
# 	    --output ${DATSET_DIR}/qrecc_provenances_contriever.jsonl \
# 	    --dense_retrieval \
# 	    --q-encoder facebook/contriever-msmarco \
# 	    --device cuda:2 \
# 	    --batch_size 54 \
# 	    --threads 4 \
# 	    --index_dir ${INDEX_DIR} \
# 	    --k 100 

# 1-3b
construct_provenances_qrecc:
	python3 src/data_augmentation/handler.py \
	    --input data/qrecc_provenances_bm25.jsonl \
	    --output data/fidqa.train.bm25.ovl.jsonl \
	    --collections ${CORPUS} \
	    --topk 100 --N 10 --overlapped
	# python3 src/data_augmentation/handler.py \
	#     --input data/qrecc_provenances_contriever.jsonl \
	#     --output data/fidqa.train.contriever.ovl.jsonl \
	#     --collections ${CORPUS} \
	#     --topk 100 --N 10 --overlapped
