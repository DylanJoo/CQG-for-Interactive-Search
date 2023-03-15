# CQG-for-Interactive-Search

This repo is for clarification question generation in interactive search (e.g., Conversational or session-baed web search).
Entire process includes

1. Construct SERP of ClariQ
2. TBD
3. TBD

---
```
Requirements

pyserini
```
---
## Construct SERP of ClariQ

### Download datasets
Download the ClairQ and CANARD dataset. The files have already stored in [data](data/).
Check the original repo [(ClarQ)](#) and [(CANARD)](#) for details

### Build lucene index 
Build the inverted index of passages in the corpus using pyserini toolkit.
Recommend to use pyserini API to build the lucene index.
```
# Build index
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
# Run `bash construct_clariq_provenanc.sh` as follow
python3 tools/retrieve_provenance.py \
    --clariq data/clariq/train.tsv \
    --output <output file (clariq + provenance)> \
    --index_dir <directory of indexes> \
    --k 100 \
    --k1 0.9 \
    --b 0.4
```

