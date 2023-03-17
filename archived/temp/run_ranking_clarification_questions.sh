export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export CUDA_VISIBLE_DEVICES=1

# p=30
# implicit clarification implicit qp
# QLM: cqg
# CQG: cqg-implicit-qp30-1004000 
# CQR: TOP30 CQE-ZS on question banks

python3 tools/cqg_qlm_ranking.py \
  --topics_with_implicit_kw data/clarifying_questions/cast22.cqg.keywords.qp30.jsonl \
  --model_path cqg/checkpoints/t5-base-cqg-implicit-qp30-1004000/ \
  --output mi_submissions/qlm_kw_cqg+dqr2.json \
  --question_bank data/clarifying_questions/retrieved_mixed_initiative_question_top5.tsv \
  --clarifications data/clarifying_questions/cast22_clarification_implicit-3qp30-1004000.tsv \
  --n_history 3

