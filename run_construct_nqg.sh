# q generation
python3 tools/convert_clariq_to_seq2seq.py  \
  --path_multiturn_jsonl data/clariq.multiturn.train.synthetic.implicit.q.jsonl \
  --path_output data/clariq.train.cqg.t5.tsv 

# only q
python3 tools/convert_clariq_to_seq2seq.py  \
  --path_multiturn_jsonl data/clariq.multiturn.train.synthetic.implicit.q.jsonl \
  --path_output data/clariq.train.cqg.implicit.q.t5.tsv \
  --keyword_based

# q with passage candidates
python3 tools/convert_clariq_to_seq2seq.py  \
  --path_multiturn_jsonl data/clariq.multiturn.train.synthetic.implicit.qp30.jsonl \
  --path_output data/clariq.train.cqg.implicit.qp30.t5.tsv \
  --keyword_based
