python3 -m pyserini.search.faiss \
  --encoder-class dpr \
  --encoder facebook/facebook/dpr-question_encoder-multiset-base \
  --topics data/clariq/train.tsv \
  --index miracl-v1.0-ar-mdpr-tied-pft-msmarco \
  --output run.miracl.mdpr-tied-pft-msmarco.ar.dev.txt \
  --batch 128 --threads 16 --hits 1000

