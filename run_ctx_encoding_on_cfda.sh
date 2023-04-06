CORPUS_PREFIX=/home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl

for i in {00000..00025};do
    python3 -m pyserini.encode input \
      --corpus ${CORPUS_PREFIX}${i} \
      --fields text \
      --shard-id 0 \
      --shard-num 1 output \
      --embeddings /home/jhju/indexes/odcqa-dpr-multiset-bases/dir-${i} \
      --to-faiss encoder \
      --encoder facebook/dpr-ctx_encoder-multiset-base \
      --fields text \
      --batch 48 \
      --fp16 \
      --device cuda:1
    echo $CORPUS_PREFIX $i "encoding and indexing is finished." >> log
done
