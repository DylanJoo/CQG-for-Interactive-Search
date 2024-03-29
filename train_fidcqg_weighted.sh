export CUDA_VISIBLE_DEVICES=1
python3 src/train_fidcqg.py \
    --model_name_or_path google/flan-t5-base \
    --config_name google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --n_contexts 10 \
    --output_dir checkpoints/fidcqg.contriever-w.ovl \
    --train_file data/fidcqg.train.contriever.ovl.jsonl \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_src_length 128 \
    --max_tgt_length 64 \
    --max_steps 4000 \
    --save_steps 1000 \
    --eval_steps 50 \
    --do_train true \
    --do_eval true \
    --learning_rate 1e-4 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --remove_unused_columns false \
    --tfidf_weighted true \
    --report_to wandb
