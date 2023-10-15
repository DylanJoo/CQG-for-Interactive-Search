# Interactive-Search

The reproduction pipeline includes these procedures

1. Construct SERP of ClariQ. (data_augmentation)
2. Fine-tune FiD-CQG model.
3. Generate (inference) the retrieval-enhanced CQ. (data_augmentation)
4. Fine-tune FiD-MRG model.

---
## Prerequisite
### Data (at CFDA4)
All the data files are located at cfda4, or you can download at [huggingface hub](https://huggingface.co/datasets/DylanJHJ/CQG/tree/main).
```
- Dataset: /home/jhju/huggingface_hub/CQG/
clariq_provenances_contriever.jsonl  
fidcqg.train.contriever.ovl.jsonl  
qrecc_provenances_contriever.jsonl
clariq_provenances_bm25.jsonl  
fidcqg.train.bm25.ovl.jsonl          
qrecc_provenances_bm25.jsonl

- Wiki's Corpus:
CORPUS_DIR=/home/jhju/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl
- Wiki's Corpus index (bm25):
INDEX_DIR=/home/jhju/indexes/wikipedia-lucene
- Corpus index (contriever): /home/jhju/indexes/wikipedia-contriever
```

You can also download the raw data and run with the codes below.
- Dataset: [ClariQ](https://github.com/aliannejadi/ClariQ) for CQG, and [QReCC](https://github.com/apple/ml-qrecc) for ConvQA.
```
wget https://github.com/aliannejadi/ClariQ/raw/master/data/train.tsv -P data/
wget https://github.com/apple/ml-qrecc/raw/main/dataset/qrecc_data.zip -P data/
unzip data/qrecc_data.zip
```
- Corpus: Wikipedia dump Dec. 20, 2018: see [Contriver's repositary](https://github.com/facebookresearch/contriever) for detail.
- Corpus indexes: you can index yourself (see 1.1) or download from [here](#).

### Requirments
```
pyserini
transformers
datasets
```

## 1 Data augmentation for FiDCQG with ClariQ
### 1-1 Build index
Build the inverted index for wiki corpus using pyserini toolkit.
For both sparse and dense retrieval, we used pyserini API and build the lucene/FAISS index.

- Lucene indexing: we append the title as part of the content.
```
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ${CORPUS_DIR} \
  --index ${INDEX_DIR} \
  --generator DefaultLuceneDocumentGenerator \
  --fields title \
  --threads 8
```
- FAISS index: we adopted the contriver's pre-built index. More detail can be found in the [repo](https://github.com/facebookresearch/contriever).

> [TODO] Try the common crawl corpus (has been downloaded)

### 1-2 Retrieve passages for ClariQ (sparse)
We demonstrate the sparse sesarch backbone. 
```
python3 src/data_augmentation/retrieve_passages.py \
    --clariq data/clariq/train.json \
    --output data/clariq_provenances_bm25.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir ${INDEX_DIR} \
    --k 100
```
You can also replace it with dense retreival.

### 1-3 Construct provenances for ClariQ/prepare training data for FiDCQG
There have many possible ways to construct the provenances.
Here, we used the **overlapped** passages as provenances. (see section 4 as well)
```
python3 src/data_augmentation/handler.py \
    --input data/clariq_provenances_bm25.jsonl \
    --output data/fidcqg.train.bm25.ovl.jsonl \
    --collections ${CORPUS} \
    --topk 100 --N 10 --overlapped
```

### 2 Fine-tune FiD-CQG
Fine-tune the FiD-T5 model with synthetic ClariQ-SERP.
I use the default training setups, you can also find more detail in `src/train_fidcqg.py` for detail.

```
python3 src/train_fidcqg.py \
    --model_name_or_path google/flan-t5-base \
    --config_name google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --n_contexts 10 \
    --output_dir checkpoints/fidcqg.bm25.ovl \
    --train_file data/fidcqg.train.bm25.ovl.jsonl \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_src_length 128 \
    --max_tgt_length 64 \
    --max_steps 4000 \
    --save_steps 1000 \
    --eval_steps 50 \
    --do_train true \
    --do_eval true \
    --learning_rate 1e-4 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --remove_unused_columns false
```

## 3 Data augmentation for FiDMRG with QReCC and FiDCQG
### 3-1 Retrieve passages for QReCC (sparse)
We first retrieved passage with standard query index, and question + answer as well.
It will take a long time, we recommend you to download the pre-retrieved data.
```
python3 src/data_augmentation/retrieve_passages.py \
    --qrecc data/qrecc/qrecc_train.json \
    --output ${DATASET_DIR}/qrecc_provenances_bm25.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir ${INDEX_DIR} \
    --k 100
```


### 3-2 Predict pseudo clarifying questions for QRecc
There have many possible ways to construct the provenances.
Here, we used the **standard** passages as provenances. (retrieve using query only)
```
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
```

### 3-3 Construct provenance for QReCC/prepare training data for FiDMRG
There have many possible ways to construct the provenances.
Here, we used the **standard** passages as provenances. (retrieve using query only)
> NOTE: try TART's negative sampling methods 
> NOTE: try considering predicteed predicted clarifying question as well

```
PRED_CQ=predictions/qrecc_cq_pred.fidcqg.bm25.ovl.jsonl
python3 src/data_augmentation/handler.py \
    --input ${DATASET_DIR}/qrecc_provenances_bm25.jsonl \
    --input_cqg_predictions ${PRED_CQ} \
    --output ${DATASET_DIR}/fidmrg.train.bm25.ovl_cqpred.bm25.ovl.jsonl \
    --collections ${CORPUS} \
    --topk 100 --N 10 --overlapped
```

### 4 Fine-tune FiD-MRG
```
python3 src/train_fidmrg.py \
    --model_name_or_path google/flan-t5-base \
    --config_name google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --n_contexts 10 \
    --output_dir checkpoints/fidmrg.bm25.ovl_dual_input \
    --train_file data/fidmrg.train.bm25.ovl_cqpred.bm25.ovl.jsonl \
    --random_sample true \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_src_length 128 \
    --max_tgt_length 64 \
    --max_steps 10000 \
    --save_steps 5000 \
    --eval_steps 500 \
    --do_train true \
    --do_eval true \
    --learning_rate 1e-4 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --remove_unused_columns false \
    --report_to wandb
```
