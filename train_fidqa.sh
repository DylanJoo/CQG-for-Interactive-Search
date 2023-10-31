# batch size = 8
# but 4 when emunerated
TRAIN_FILE=data/fidmrg.train.bm25.w.ovl_cqpred.bm25.ovl.jsonl
export CUDA_VISIBLE_DEVICES=0
python3 src/train_fidmrg.py \
    --model_name_or_path google/flan-t5-base \
    --config_name google/flan-t5-base \
    --tokenizer_name google/flan-t5-base \
    --n_contexts 10 \
    --output_dir checkpoints/fidmrg_answer.bm25.fidcqg-w \
    --train_file $TRAIN_FILE \
    --random_sample false \
    --enumerated_sample false \
    --answer_sample true \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_src_length 128 \
    --max_tgt_length 64 \
    --max_steps 10000 \
    --save_steps 5000 \
    --eval_steps 500 \
    --do_train true \
    --do_eval true \
    --learning_rate 1e-4 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --remove_unused_columns false \
    --report_to wandb

