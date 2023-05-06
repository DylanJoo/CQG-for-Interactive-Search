mkdir -p evaluation
mkdir -p evaluation/data

## (1) retrieve K provenances
## Lucene index using contents with title (title content)
python3 src/pre/retrieve_passages2.py \
    --clariq data/clariq/dev.tsv \
    --output evaluation/data/clariq_provenances_lucene.jsonl \
    --k1 0.9 --b 0.4 \
    --index_dir /home/jhju/indexes/full_wiki_segments_lucene \
    --k 100

### FAISS index using contents with title (title [SEP] content)
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
for indexing in lucene dpr;do
    python3 src/pre/organize_provenances.py \
        --questions_with_provenances evaluation/data/clariq_provenances_dpr.jsonl \
        --collections /home/jhju/datasets/full_wiki_segments/full_wiki_segments.jsonl \
        --output evaluation/data/dev_fidcqg_v0_${indexing}.jsonl \
        --N 10 \
        --topk 100
done
        # --random

## (3) Generate clarification questions
for model in fidcqg*;do
    # for version in v0_rand20;do
    for version in v0_lucene v0_dpr;do
        ckpt=4000
        python3 src/inference_fidcqg.py \
            --jsonl_file evaluation/data/dev_fidcqg_${version}.jsonl \
            --output_file evaluation/cqg/dev_${model}-${ckpt}-${version}_pred.jsonl \
            --batch_size 4 \
            --used_checkpoint ${model}/checkpoint-${ckpt}/ \
            --used_tokenizer t5-base \
            --calculate_crossattention \
            --n_contexts 10 \
            --max_length 20 \
            --device 'cuda:2' \
            --do_sample \
            --top_k 10
        python3 src/tools/convert_jsonl_to_txt.py \
            --path_jsonl evaluation/cqg/dev_${model}-${ckpt}-${version}_pred.jsonl \
            --path_txt evaluation/cqg/dev_${model}-${version}_pred.txt
    done
done
