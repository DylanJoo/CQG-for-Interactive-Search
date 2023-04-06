for i in 0..250;do
    python -m pyserini.encode input \
      --corpus /content/full_wiki_segments.jsonl \
      --fields text \
      --delimiter "\n" \
      --shard-id $i \
      --shard-num 250 output \
      --embeddings odcqa-dpr-multiset-bases/dir-$i \
      --to-faiss encoder \
      --encoder facebook/dpr-ctx_encoder-multiset-base \
      --fields text \
      --batch 32 \
      --fp16
done
