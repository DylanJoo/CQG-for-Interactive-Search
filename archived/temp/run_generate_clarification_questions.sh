export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export CUDA_VISIBLE_DEVICES=1


# (1) Contruct the keyword set of implicit user needs
python3 tools/cqg_kw_extraction.py \
  -k_docs 30 -k1 0.82 -b 0.68 -k_keywords 10 \
  -index /tmp2/trec/cast/indexes/cast22_doc \
  -query data/parsed_2022_evaluation_topics_v1.0.jsonl \
  -output data/clarifying_questions/cast22.cqg.keywords.qp30.jsonl

# (2) Clarification question generation from t5-cqg-implicit-qp
python3 tools/cqg_generation.py \
  --topics_with_implicit_kw data/clarifying_questions/cast22.cqg.implicit.p30.jsonl \
  --model_path cqg/checkpoints/t5-base-cqg-implicit-qp30-1004000/ \
  --output data/clarifying_questions/cast22_clarification_implicit-3qp30-1004000.tsv \
  --n_history 3

