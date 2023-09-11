CORPUS_DIR=~/datasets/wiki.dump.20181220/
CORPUS=${CORPUS_DIR}/wiki_psgs_w100.jsonl

preprocess_wiki_corpus:
	python3 src/tools/tsv_to_jsonl.py \
	  --path_tsv ${CORPUS_DIR}/wiki_psgs_w100.tsv \
	  --path_jsonl ${CORPUS_DIR}/wiki_psgs_w100.jsonl

index_wiki_corpus_bm25:
	python3 -m pyserini.index.lucene \
	  --collection JsonCollection \
	  --input ${CORPUS_DIR} \
	  --index ${INDEX_DIR} \
	  --generator DefaultLuceneDocumentGenerator \
	  --fields title \
	  --threads 8

construct_serp_clariq:
	# python3 src/data_augmentation/retrieve_passages.py \
	#     --clariq data/clariq/train.tsv \
	#     --output data/clariq_provenances_bm25.jsonl \
	#     --k1 0.9 --b 0.4 \
	#     --index_dir ${INDEX_DIR} \
	#     --k 100
	# contriever
	python3 src/data_augmentation/retrieve_passages.py \
	    --clariq data/clariq/train.tsv \
	    --output data/clariq_provenances_contriever.jsonl \
	    --dense_retrieval \
	    --q-encoder facebook/contriever-msmarco \
	    --device cuda:2 \
	    --batch_size 32 \
	    --threads 4 \
	    --index_dir ${INDEX_DIR} \
	    --k 100 

select_serp_clariq:
	python3 src/data_augmentation/handler.py \
	    --input data/clariq_provenances_bm25.jsonl \
	    --output data/fidcqg.train.bm25.ovl.jsonl \
	    --collections ${CORPUS} \
	    --topk 100 --N 10 --overlapped
	# contriever
	# python3 src/data_augmentation/handler.py \
	#     --input data/clariq_provenances_contriever.jsonl \
	#     --output data/fidcqg.train.bm25.ovl.jsonl \
	#     --collections ${CORPUS} \
	#     --topk 100 --N 10 --overlapped

construct_serp_qrecc:
	python3 src/data_augmentation/retrieve_passages.py \
	    --qrecc data/qrecc/qrecc_train.json \
	    --output data/qrecc_provenances_bm25.jsonl \
	    --k1 0.9 --b 0.4 \
	    --index_dir ${INDEX_DIR} \
	    --k 100
	# dense retrieval
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
