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

- Corpus: /home/jhju/datasets/wiki.dump.20181220/wiki_psgs_w100.jsonl
- Corpus index (bm25): /home/jhju/indexes/wikipedia-lucene
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

### 1 Construct SERP for ClariQ
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
    --clariq data/clariq/train.json \
    --output data/clariq_provenances_bm25.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir ${INDEX_DIR} \
    --k 100
```
You can also replace it with dense retreival.

### 1-2b Retrieve passages for QReCC
It will take a long time, we recommend you to download the pre-retrieved data.
```
python3 src/data_augmentation/retrieve_passages.py \
    --qrecc data/qrecc/qrecc_train.json \
    --output data/qrecc_provenances_bm25.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir ${INDEX_DIR} \
    --k 100
```

### 1-3 Construct provenances (create training data) for ClariQ
There have many possible ways to construct the provenances.
Here, we used the **overlapped** passages as provenances. (see `handler.py` for detail)
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
    --model_name_or_path t5-base \
    --n_contexts 10 \
    --output_dir ./fidcqg \
    --train_file data/fid.train.bm25.ovl.jsonl \ 
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

### 3-1 Construct provenances (create trianing data) for QRecc
# Predict clarification questions for QReCC queries.
# Collect the target of miresponse: answers and c_question
```
TBD
```

### 3-2 Fine-tune FiD-MRG
```
TBD
```

## 4. 
