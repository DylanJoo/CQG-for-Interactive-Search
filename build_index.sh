# indexing for wiki corpus
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /tmp2/jhju/datasets/odqa-psgs/ \
  --index /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_contents \
  --generator DefaultLuceneDocumentGenerator \
  --threads 4 \
  --storePositions --storeDocvectors --storeRaw
