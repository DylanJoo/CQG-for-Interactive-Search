# ClariQ
mkdir -p data/clariq

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

