# [Convert to pyserini format]
# python3 src/tools/convert_tsv_to_jsonl.py \
#   --path_tsv /home/jhju/datasets/full_wiki_segments.tsv  \
#   --path_jsonl /home/jhju/datasets/full_wiki_segments/tc/full_wiki_segments.jsonl \
#   --add_title 

# indexing for wiki corpus (contents + title)
# python3 -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input /tmp2/jhju/datasets/odqa-psgs/tc/ \
#   --index /tmp2/jhju/indexes/odqa-psgs_w100_jsonl_tc \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 4 \
#   --storeDocvectors 

