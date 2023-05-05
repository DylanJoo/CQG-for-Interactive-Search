mkdir -p evaluation
mkdir -p evaluation/data

## (1) retrieve K provenances
# python3 src/pre/retrieve_passages2.py \
#     --clariq data/clariq/dev.tsv \
#     --output evaluation/data/clariq_provenances_dpr.jsonl \
#     --dense_retrieval \
#     --q-encoder facebook/dpr-question_encoder-multiset-base \
#     --device cuda:2 \
#     --batch_size 32 \
#     --threads 4 \
#     --index_dir /home/jhju/indexes/full_wiki_segments_dpr \
#     --k 100 

## (2) set the provenances for FiD
# python3 src/pre/organize_provenances.py \
#     --questions_with_provenances evaluation/data/clariq_provenances_dpr.jsonl \
#     --collections /home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl \
#     --output evaluation/data/dev_fidcqg_v0.jsonl \
#     --N 10 \
#     --topk 100

# python3 src/pre/organize_provenances.py \
#     --questions_with_provenances evaluation/data/clariq_provenances_dpr.jsonl \
#     --collections /home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl \
#     --output evaluation/data/dev_fidcqg_v0_rand20.jsonl \
#     --N 10 \
#     --topk 20 \
#     --random

## (3) Generate clarification questions
for model in fidcqg*;do
    for version in v0 v0_rand20;do
        ckpt=2000
        python3 src/inference_fidcqg.py \
            --jsonl_file evaluation/data/dev_fidcqg_${version}.jsonl \
            --output_file evaluation/cqg/dev_${model}-${ckpt}-${version}_pred.jsonl \
            --batch_size 8 \
            --used_checkpoint ${model}/checkpoint-${ckpt}/ \
            --used_tokenizer t5-base \
            --calculate_crossattention \
            --n_contexts 10 \
            --max_length 256 \
            --device 'cuda:2'
    done
done
