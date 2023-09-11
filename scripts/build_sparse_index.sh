# python3 src/tools/convert_tsv_to_jsonl.py \
#   --path_tsv /home/jhju/datasets/full_wiki_segments.tsv  \
#   --path_jsonl /home/jhju/datasets/full_wiki_segments/tc/full_wiki_segments.jsonl \
#   --add_title 

# Wiki corpus (contents + title)
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /home/jhju/datasets/full_wiki_segments/tmpe \
  --index /home/jhju/indexes/full_wiki_segments_lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storeDocvectors 
