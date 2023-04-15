CORPUS_PREFIX=/home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl
OUTPUT_PREFIX=/home/jhju/indexes/odcqa-dpr-multiset-bases/dir-

echo "-------\n" >> log
echo "Run the following shards..." >> log

for i in {00000..00025};do
    if [ -f "${OUTPUT_PREFIX}${i}"/success ]; then
        echo ${OUTPUT_PREFIX}${i} "exists and has been done"
    else
        echo ${OUTPUT_PREFIX}${i} "does not exist."
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
        touch ${OUTPUT_PREFIX}${i}/success
        echo "Shard" $i  >> log
    fi
done
