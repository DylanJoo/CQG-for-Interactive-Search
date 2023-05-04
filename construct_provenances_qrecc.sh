## (1) retrieve K provenances
# python3 src/pre/retrieve_passages.py \
#     --qrecc data/qrecc/qrecc_train.json \
#     --output data/qrecc_provenances_tc.jsonl \
#     --index_dir /tmp2/jhju/indexes/odcqa-psgs \
#     --k 100 \
#     --k1 0.9 \
#     --b 0.4
python3 src/pre/retrieve_passages2.py \
    --qrecc data/qrecc/qrecc_train.json \
    --output data/qrecc_provenances_dpr.jsonl \
    --dense_retrieval \
    --q-encoder facebook/dpr-question_encoder-multiset-base \
    --device cuda \
    --batch_size 32 \
    --threads 8 \
    --index_dir /home/jhju/indexes/full_wiki_segments_dpr \
    --k 100 

## (2) set the provenances for FiD
# python3 src/pre/organize_provenances.py \
#     --questions_with_provenances data/qrecc_provenances_tc.jsonl \
#     --collections /tmp2/jhju/datasets/odcqa-psgs/full_wiki_segments.jsonl \
#     --output data/qrecc_qa_provenances.jsonl \
#     --N 10 \
#     --exclusive

## (3) Generate clarification questions
# python3 src/inference_fidcqg.py \
#     --jsonl_file data/qrecc_qa_provenances.jsonl \
#     --output_file data/qrecc_cquestions.jsonl \
#     --batch_size 4 \
#     --used_checkpoint fidcqg/checkpoint-10000/ \
#     --used_tokenizer t5-base \
#     --calculate_crossattention \
#     --n_contexts 10 \
#     --max_length 256 \
#     --device 'cuda'

# (4) Merge the 'predicted' clarification questions
# python3 src/pre/organize_miresponses.py \
#     --convqa data/qrecc_qa_provenances.jsonl \
#     --convqcq data/qrecc_cquestions.jsonl \
#     --output data/train_fidmrg_v0.jsonl
