export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Build index without stemming
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /tmp2/jhju/datasets/cqg/ \
  --index /tmp2/jhju/indexes/msmarco-psgs/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9 \
  --stemmer none --storeDocvectors 
