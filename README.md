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
[ClariQ](https://github.com/aliannejadi/ClariQ) and 
[QReCC](https://github.com/apple/ml-qrecc).
Note that QReCC contains [QuAC](https://sites.google.com/view/qanta/projects/canard), [CANARD](https://sites.google.com/view/qanta/projects/canard) and TRECCAsT dataset.

- Corpus: 
We use the wiki dump with the preprocessed codes in DPR. The preprocessed corpus has about 21M passages with title. 
The scripts are from original [DPR repo](#).

In case you would like to modify the data; the following scripts provide the detail of data and preprocessing and formatting.
```
bash download_data.sh

# train data
python3 src/tools/parse_canard.py \
  --path_canard data/canard/train.json \
  --path_output data/canard/train.jsonl \
  --dir_quac data/quac/
```
### Packages
```
pyserini
transformers 
datasets
```
---

## 1. Construct SERP of ClariQ

### Build lucene index 
Build the inverted index of passages in the corpus using pyserini toolkit.
Recommend to use pyserini API to build the lucene index.
```
# Run `build_index.sh`
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input <directory of jsonl> \
  --index <directory of indexes> \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw
```

### Construct provenance aka SERP (wiki passage excerpts) for each questions
You can also run `construct_provenance_clariq.sh`. 
This code includes two steps.

(a) Search from the corpus and retrieve top-K passages via BM25 search.
```
python3 src/pre/retrieve_passages.py \
    --clariq data/clariq/train.tsv \
    --output <CLARIQ_PRV> \
    --index_dir <INDEX_DIR> \
    --k 100 \
    --k1 0.9 \
    --b 0.4
```

(b) Select N passages and prepare the CQG inputs. This data is the training data for FiD-CQG.
```
python3 src/pre/organize_provenances.py \
    --questions_with_provenances data/clariq_provenances_tc.jsonl \
    --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
    --output data/train_fidcqg_v0.jsonl \
    --N 10
```

## 2. Fine-tune FiD-CQG: Corpus-aware clarification question generation
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
You can also run `construct_provenances_canard.sh`, which is the same as the following codes.

### Retrieve provenances (wiki passage excerpts) for each ConvQA's questions
We use CANARD's rewritten query and retrieve top-K passages.
Follow the stage 1: we use BM25 for searching topK passages from the corpus.
```
# Find K provenance candidates
python3 src/pre/retrieve_passages.py \
    --canard data/canard/train.jsonl \
    --collections None \
    --output data/<CANARD_PROV> \
    --index_dir <INDEX_DIR>
    --k 100 \
    --k1 0.9 \
    --b 0.4

# Select N provenances
python3 src/pre/organize_provenances.py \
    --questions_with_provenances data/<CANARD_PROV> \
    --collections /tmp2/jhju/datasets/odqa-psgs/psgs_w100.jsonl \
    --output data/test_canard_fidcqg.jsonl \
    --N 10
```

### Inference FiD-CQG with CANARD queries: Generate clarification questions
Run text generation script `inference_fidcqg.py` with the fine-tuned checkpoint as our default settings.
This script will generate the corresponding clarification questions.
Follow this `jsonl` data format.
```
# Inference predicted CQ
python3 src/inference_fidcqg.py \
    --jsonl_file test_canard_fidcqg.jsonl \
    --output_file <CANARD_CQG> \
    --used_checkpoint ./fidcqg/checkpoint-10000/ \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 50 \
    --device 'cuda' \
```
Then, prepare the MRG training instances, include: (1) Clarificiation questions(Predicted) and (2) Answer (from QuAC).
```
```



## 4. 
