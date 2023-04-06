CORPUS_PREFIX=/content/full_wiki_segments/full_wiki_segments.jsonl
OUTPUT_PREFIX=/content/odcqa-dpr-multiset-bases/dir-

echo "-------\n" >> log
echo "Run the following shards..." >> log

for i in {00025..00000};do
    python -m pyserini.encode input \
      --corpus ${CORPUS_PREFIX}${i} \
      --fields text \
      --shard-id 0 \
      --shard-num 1 output \
      --embeddings ${OUTPUT_PREFIX}${i} \
      --to-faiss encoder \
      --encoder facebook/dpr-ctx_encoder-multiset-base \
      --fields text \
      --batch 32 \
      --fp16
    echo "Shard" $i  >> log
    touch ${OUTPUT_PREFIX}${i}/success
    tar cvf /content/odcqa-dpr-multiset-bases-dir-${i}.tar /content/odcqa-dpr-multiset-bases/dir-${i}
done


