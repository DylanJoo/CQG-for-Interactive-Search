# ClariQ
mkdir -p data/clariq

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

