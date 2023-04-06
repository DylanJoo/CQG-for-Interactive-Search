CORPUS_PREFIX=/content/full_wiki_segments/full_wiki_segments.jsonl
echo "-------\n" >> log
echo "Run the following shards..." >> log
for i in {00025..00000};do
    python -m pyserini.encode input \
      --corpus ${CORPUS_PREFIX}${i} \
      --fields text \
      --shard-id 0 \
      --shard-num 1 output \
      --embeddings /content/odcqa-dpr-multiset-bases/dir-${i} \
      --to-faiss encoder \
      --encoder facebook/dpr-ctx_encoder-multiset-base \
      --fields text \
      --batch 32 \
      --fp16
    echo "Shard" $i  >> log
done

