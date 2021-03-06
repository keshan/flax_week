export MODEL_DIR=Sinhala-roberta
./run_mlm_flax.py \
	--output_dir="${MODEL_DIR}" \
	--model_type="roberta" \
	--config_name="${MODEL_DIR}" \
	--tokenizer_name="${MODEL_DIR}" \
	--dataset_name="keshan/clean-si-mc4" \
	--dataset_config_name="si" \
	--max_seq_length="128" \
	--weight_decay="0.01" \
	--per_device_train_batch_size="128" \
	--per_device_eval_batch_size="128" \
	--learning_rate="3e-4" \
	--warmup_steps="1000" \
	--overwrite_output_dir \
	--num_train_epochs="18" \
	--adam_beta1="0.9" \
	--adam_beta2="0.98" \
	--logging_steps="500" \
	--save_steps="2500" \
	--eval_steps="2500" \
	--push_to_hub

