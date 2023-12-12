# topiocqa
# "data.topiocqa_dataset.train"
wget https://zenodo.org/records/7709644/files/topiocqa_train.json
# "data.topiocqa_dataset.dev"
wget https://zenodo.org/records/7709644/files/topiocqa_dev.json
# "data.wikipedia_split.full_wiki"
wget https://zenodo.org/records/6173228/files/data/wikipedia_split/full_wiki.jsonl
# data.wikipedia_split.full_wiki_segments
wget https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv

# passage_embeddings.all_history.wikipedia_passages
for i in 0..50;do
    wget https://zenodo.org/records/6153453/files/passage_embeddings/all_history/wikipedia_passages_$i.pkl
done
