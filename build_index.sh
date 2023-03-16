# indexing for wiki corpus (contents)
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /tmp2/jhju/datasets/odqa-psgs/ \
  --index /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_c \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw &

# indexing for wiki corpus (contents + title)
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /tmp2/jhju/datasets/odqa-psgs/tc/ \
  --index /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw
