export MODEL_DIR=wav2vec2-si-base

./run_wav2vec2_pretrain_flax.py \
	--output_dir=${MODEL_DIR} \
    --dataset_name="keshan/large-sinhala-asr-dataset" \
    --validation_split_name="test" \
	--num_train_epochs="5" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--learning_rate="5e-4" \
	--weight_decay="0.01" \
	--warmup_steps="2000" \
	--model_name_or_path=${MODEL_DIR} \
	--preprocessing_num_workers="64" \
	--max_duration_in_seconds="10.0" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--pad_to_multiple_of="16384" \
	--push_to_hub
