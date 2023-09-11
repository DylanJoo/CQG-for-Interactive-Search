# Interactive-Search

The reproduction pipeline includes these procedures

1. Construct SERP of ClariQ. (pre)
2. Fine-tune FiD-CQG model.
3. Generate (inference) the retrieval-enhanced CQ.
4. Fine-tune FiD-MRG model.

---
## Prerequisite
### Data
Datasets:
* Clarification question generation: [ClariQ](https://github.com/aliannejadi/ClariQ)
* Open-domain conversational QA: [QReCC](https://github.com/apple/ml-qrecc).

Corpus:
* Wikipedia dump Dec. 20, 2018: see [Contriver's repositary](https://github.com/facebookresearch/contriever) for detail.

### Requirments
```
pyserini=??
transformers=??
datasets=??
```

### 1. Construct SERP for ClariQ and QReCC

As SERP for qrecc is similar to the retrieval for clariq, we only show the scripts for clariq. 
The scripts for qrecc can be found in the `Makefile` 

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

### 1-2a Retrieve passages for ClariQ
We demonstrate the sparse sesarch backbone. 
```
python3 src/data_augmentation/retrieve_passages.py \
    --qrecc data/clariq/clariq_train.json \
    --output data/clariq_provenances_bm25.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir ${INDEX_DIR} \
    --k 100
```

You can also replace it with dense retreival with this scripts:
```
python3 src/pre/retrieve_passages2.py \
    --clariq data/clariq/train.tsv \
    --output data/clariq_provenances_contriever.jsonl \
    --index_dir ${INDEX_DIR} \
    --threads 8 \
    --k 100 \
    --dense_retrieval \
    --q-encoder facebook/contriever-msmarco \
    --device cuda \
    --batch_size 32
```

### 1-2b Retrieve passages for qrecc
It will take a long time, we recommend you to download the pre-retrieved data stored [here](#).
```
python3 src/data_augmentation/retrieve_passages.py \
    --qrecc data/qrecc/qrecc_train.json \
    --output data/qrecc_provenances_bm25.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir ${INDEX_DIR} \
    --k 100
```

### 2. Fine-tune FiD-CQG: Corpus-aware clarification question generation
Fine-tune the FiD-T5 model with synthetic ClariQ-SERP.
I use the default training setups, you can also find more detail in `src/train_fidcqg.py` for detail.
```
# Run `train_fidcqg.py` 
python3 src/train_fidcqg.py \
    --model_name_or_path t5-base \
    --n_contexts 10 \
    --output_dir ./fidcqg \
    --train_file <TRAIN_CQG> \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --max_length 256 \
    --max_steps 100 \
    --save_steps 50\
    --eval_steps 25 \
    --do_train true \
    --do_eval false \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --remove_unused_columns false
```

## 3. End-to-end Training data augmentation: generate clarification questions

You can also run `construct_provenances_canard.sh`, which is the same as the following four steps.

```
# Find K provenance candidates
python3 src/pre/retrieve_passages.py \
    --qrecc data/qrecc/train.??? \
    --output data/<QRECC_PROV> \
    --index_dir <INDEX_DIR>
    --k 100 \
    --k1 0.9 \
    --b 0.4

# Select N provenances
python3 src/pre/organize_provenances.py \
    --questions_with_provenances data/<CANARD_PROV> \
    --collections <CORPUS_PATH (title-separated)> \
    --output data/test_canard_fidcqg.jsonl \
    --N 10

# Predict clarification questions for QReCC queries.
# Use the fine-tuned FiD-CQG model.
python3 src/inference_fidcqg.py \
    --jsonl_file test_canard_fidcqg.jsonl \
    --output_file <QRECC_CQG> \
    --used_checkpoint ./fidcqg/checkpoint-10000/ \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 50 \
    --device 'cuda' \

# Rearrage the target of miresponse, including questions and c_question
# [TODO] such sampling method can be further improved.
python3 src/pre/organize_miresponses.py \
    --convqa data/qrecc/train.??? \
    --convcqa <QRECC_CQG> \
    --output data/train_fidmrg_v0.jsonl 
```

## 4. 
