# CQG-for-Interactive-Search

This repo is for clarification question generation in interactive search (e.g., Conversational or session-baed web search).
Entire process includes

1. Construct SERP of ClariQ.
2. Fine-tune FiD-CQG model.
3. Generate (inference) the CQ for CANARD dataste.
4. TBD

---
## Prerequisite
There are some data and pacakge you need to install.

### Data
- Datasets:
Download the ClairQ and CANARD dataset.  The files have already stored in [data](data/).
Check the original page 
[ClariQ](https://github.com/aliannejadi/ClariQ), 
[QuAC](https://sites.google.com/view/qanta/projects/canard), and
[CANARD](https://sites.google.com/view/qanta/projects/canard).
for detail.

- Corpus: 
We use the wiki dump with the preprocessed codes in DPR. The preprocessed corpus has about 21M passages with title. 
The scripts are from original [DPR repo](#).

You can ruTBD
```
bash download_data.sh
```
### Packages
```
pyserini
transformers 
datasets
```
---

## Construct SERP of ClariQ

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
Search from the corpus and retrieve top-K passages via BM25 search.
```
# Run `construct_provenance_clariq.sh`
python3 src/pre/retrieve_passages.py \
    --clariq data/clariq/train.tsv \
    --output <output file> \
    --index_dir <directory of indexes> \
    --k 100 \
    --k1 0.9 \
    --b 0.4
```

Then, select N passages and prepare the CQG inputs. This data is the training data for FiD-CQG.
```
# Run `construct_provenance_clariq.sh`
python3 src/pre/organize_provenances.py \
    --clariq data/clariq/train.tsv \
    --output <output file> \
    --index_dir <directory of indexes> \
    --k 100 \
    --k1 0.9 \
    --b 0.4
```

## Fine-tune FiD-CQG: Corpus-aware clarification question generation
Fine-tune the FiD-T5 model with synthetic ClariQ-SERP.
I use the default training setups, you can also find more detail in `src/train_fidcqg.py` for detail.
```
# Run `train_fidcqg.py` 
python3 src/train_fidcqg.py \
    --model_name_or_path t5-small \
    --n_contexts 10 \
    --output_dir ./fidcqg \
    --train_file <output file> \
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

## End-to-end Training data augmentation: generate clarification questions

### Retrieve provenances (wiki passage excerpts) for each ConvQA's questions
Search from the corpus and retrieve top-K passages via BM25 search.
Thereare several settings that can adopt for this purpose.

### Inference FiD-CQG with CANARD queries: Generate clarification questions
Run text generation script `inference_fidcqg.py` with the fine-tuned checkpoint as our default settings.
This script will generate the corresponding clarification questions.
Follow this `jsonl` data format 
```
# Run `inference_fidcqg.py` 
python3 src/inference_fidcqg.py \
    --jsonl_file <input jsonl file> \
    --output_file <output jsonl file> \
    --used_checkpoint ./fidcqg/checkpoint-10000/ \
    --calculate_crossattention \
    --n_contexts 10 \
    --max_length 50 \
    --device 'cuda' \
```
