export CUDA_VISIBLE_DEVICES=0
python3 src/train_fidcqg.py \
    --model_name_or_path t5-base \
    --config_name t5-base \
    --tokenizer_name t5-base \
    --n_contexts 10 \
    --output_dir fidcqg_dpr_weighted \
    --train_file ./data/train_fidcqg_v0.jsonl \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 128 \
    --max_steps 4000 \
    --save_steps 1000 \
    --eval_steps 200 \
    --do_train true \
    --do_eval false \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --remove_unused_columns false \
    --tfidf_weighted true
    # --tfidf_weighted_stopwords english 
