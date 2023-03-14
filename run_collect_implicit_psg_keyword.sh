export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# shard into 3
python3 tools/candidate_passage_implicit_expansion.py \
  -k_docs 30 -k1 0.82 -b 0.68 -k_keywords 10 \
  -index /tmp2/jhju/indexes/msmarco-psgs \
  -query data/clariq.multiturn.train.synthetic.implicit.q.jsonlaa \
  -output data/clariq.multiturn.train.synthetic.implicit.qp30.v2.jsonlaa &
    
python3 tools/candidate_passage_implicit_expansion.py \
  -k_docs 30 -k1 0.82 -b 0.68 -k_keywords 10 \
  -index /tmp2/jhju/indexes/msmarco-psgs \
  -query data/clariq.multiturn.train.synthetic.implicit.q.jsonlab \
  -output data/clariq.multiturn.train.synthetic.implicit.qp30.v2.jsonlab &

python3 tools/candidate_passage_implicit_expansion.py \
  -k_docs 30 -k1 0.82 -b 0.68 -k_keywords 10 \
  -index /tmp2/jhju/indexes/msmarco-psgs \
  -query data/clariq.multiturn.train.synthetic.implicit.q.jsonlac \
  -output data/clariq.multiturn.train.synthetic.implicit.qp30.v2.jsonlac &

python3 tools/candidate_passage_implicit_expansion.py \
  -k_docs 30 -k1 0.82 -b 0.68 -k_keywords 10 \
  -index /tmp2/jhju/indexes/msmarco-psgs \
  -query data/clariq.multiturn.train.synthetic.implicit.q.jsonlad \
  -output data/clariq.multiturn.train.synthetic.implicit.qp30.v2.jsonlad
