# Corpus
# refer to: https://github.com/McGill-NLP/topiocqa/blob/main/download_data.py
mkdir -p data/corpus
wget https://zenodo.org/record/6149599/files/data/wikipedia_split/full_wiki_segments.tsv -O data/corpus/full_wiki_segments.tsv
## corpus preprocess

# ClariQ
mkdir -p data/clariq
wget https://github.com/aliannejadi/ClariQ/raw/master/data/train.tsv -O data/clariq/train.tsv
wget https://github.com/aliannejadi/ClariQ/raw/master/data/dev.tsv -O data/clariq/dev.tsv

# QReCC
mkdir -p data/qrecc
wget https://github.com/apple/ml-qrecc/raw/main/dataset/qrecc_data.zip -O data/qrecc/qrecc_data.zip
unzip data/qrecc/qrecc_data.zip

# QuAC
mkdir -p data/quac
wget https://s3.amazonaws.com/my89public/quac/train_v0.2.json -O data/quac/train_v0.2.json
wget https://s3.amazonaws.com/my89public/quac/val_v0.2.json -O data/quac/val_v0.2.json

# CANARD
mkdir -p data/canard
wget https://obj.umiacs.umd.edu/elgohary/CANARD_Release.zip
unzip CANARD_Release.zip 
mv CANARD_Release/*.json data/canard/
rm CANARD_Release.zip
rm -r CANARD_Release/
rm -r __MACOSX/

