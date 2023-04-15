# CQG-for-Interactive-Search

This repo is for clarification question generation in interactive search (e.g., Conversational or session-baed web search).
Entire process includes

1. Construct SERP of ClariQ. (pre)
2. Fine-tune FiD-CQG model.
3. Generate (inference) the CQ for CANARD dataste.
4. FIne-tune FiD-MRG model.

---
## Prerequisite
There are some data and pacakge you need to install.

### Data
- Datasets:
Download the ClairQ and QReCC dataset dataset.  
The files have already stored in [data](data/).

Check the original page 
- [ClariQ](https://github.com/aliannejadi/ClariQ)
- [QReCC](https://github.com/apple/ml-qrecc).
(Note that QReCC contains [QuAC](https://sites.google.com/view/qanta/projects/canard), [CANARD](https://sites.google.com/view/qanta/projects/canard) and TRECCAsT dataset).

### Parse QReCC In case you would like to modify the data; the following scripts provide the detail of data and preprocessing and formatting.
```
# [TODO] revise this into qrecc
# [TODO] documentation update
python3 src/tools/parse_qrecc.py \
  --qrecc data/qrecc/train.?? \
  --output data/qrecc/train.?? \
  --quac data/quac/
```

- Corpus: 
We use the wiki dump with the preprocessed codes in DPR. The preprocessed corpus has about 21M passages with title. 
The scripts are from original [DPR repo](#).

### Packages
```
pyserini
transformers 
datasets
```
---

### 1. Construct SERP of ClariQ

### 1.0 Build lucene index 
Build the inverted index of passages in the corpus using pyserini toolkit.
Recommend to use pyserini API to build the FAISS or lucene index.
```
# Example of the Lucene index (`build_sparse_index.sh`).
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input <CORPUS_DIR (title included jsonl) > \
  --index <INDEX_DIR> \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw

# Example of the FAISS index (`build_dense_index.sh`).
# Note that we used the DPR context and query encoder for dense retrieval.
python3 -m pyserini.encode input \
  --corpus <CORPUS_PATH (title-included with [SEP]> \
  --fields text \
  --shard-id 0 \
  --shard-num 1 output \
  --embeddings <INDEX_DIR> \
  --to-faiss encoder \
  --encoder facebook/dpr-ctx_encoder-multiset-base \
  --fields text \
  --batch 48 \
  --fp16 \
  --device cuda:0
```

You can run `construct_provenance_clariq.sh` directly, which includes the following two steps:
```
# Find K provenance candidates
python3 src/pre/retrieve_passages.py \
    --clariq data/clariq/train.tsv \
    --output <CLARIQ_PRV> \
    --index_dir <INDEX_DIR> \
    --k 100 \
    --k1 0.9 \
    --b 0.4

# Select N provenances. This data is the training data for FiD-CQG.
```
python3 src/pre/organize_provenances.py \
    --questions_with_provenances data/clariq_provenances_tc.jsonl \
    --collections <CORPUS_DIR (title-separated jsonl) > \
    --output data/train_fidcqg_v0.jsonl \
    --N 10
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
