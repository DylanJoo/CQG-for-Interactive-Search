# CQG-for-Interactive-Search

This repo is for clarification question generation in interactive search (e.g., Conversational or session-baed web search).

---
## Building lucene index 
Build the inverted index of passages in the corpus using pyserini toolkit.
```
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /tmp2/jhju/datasets/cqg/ \
  --index /tmp2/jhju/indexes/msmarco-psgs/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9 \
  --stemmer none --storeDocvectors 
```
